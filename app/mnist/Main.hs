{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE DuplicateRecordFields #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module Main (main) where

import Control.Applicative (optional, (<**>))
import Control.Arrow
import Control.Exception (evaluate)
import qualified Control.Foldl as L
import Control.Lens hiding (Snoc, (:>))
import Control.Monad (when)
import Control.Monad.IO.Class
import Control.Monad.Trans.Resource (MonadResource, ResourceT, runResourceT)
import Control.Subcategory.Linear (unsafeToMat)
import qualified Data.ByteString as BS
import qualified Data.DList as DL
import Data.Foldable (foldlM)
import Data.Functor.Of (Of (..))
import Data.Massiv.Array (PrimMonad, Sz (..))
import qualified Data.Massiv.Array as M
import Data.Massiv.Array.IO (readImageAuto)
import Data.Maybe (fromMaybe)
import Data.Monoid (Sum (..))
import qualified Data.Persist as Persist
import Data.Strict (Pair (..))
import Data.Strict.Tuple ((:!:))
import Data.Time (defaultTimeLocale, formatTime, getZonedTime)
import qualified Data.Vector.Unboxed as U
import DeepLearning.MNIST
import DeepLearning.NeuralNetowrk.Massiv hiding (Train, scale)
import GHC.TypeNats
import Generic.Data
import qualified Options.Applicative as Opts
import qualified Streaming as SS
import qualified Streaming.ByteString as Q
import Streaming.Prelude (Stream)
import qualified Streaming.Prelude as S
import Streaming.Shuffle
import System.FilePath ((</>))
import System.IO (BufferMode (LineBuffering), hSetBuffering, stdout)
import System.Random.Stateful (RandomGenM, globalStdGen)
import Text.Printf (printf)
import UnliftIO (finally, mask_)
import UnliftIO.Directory (copyFile, createDirectoryIfMissing, doesFileExist)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  Opts.execParser cmdP >>= \case
    Train opts -> doTrain globalStdGen opts
    Recognise opts -> recognise opts

batchedName :: FilePath
batchedName = "batched.dat"

recognise :: RecognitionOpts -> IO ()
recognise RecognitionOpts {..} = do
  let inFile = modelDir </> batchedName
  putStrLn $ "Recognising: " <> inFile
  batchedNet <- readNetworkFile @BatchedNet inFile

  image <- fromGrayscaleImage <$> readImageAuto input
  when (M.size image /= Sz2 28 28) $
    error "Input image must be 28x28 pixels."
  let inp = toMNISTInput @PixelSize $ unsafeToMat image
  d <- evaluate $ predict batchedNet inp
  putStrLn $ "* Digit recognised by batchnormed NN: " <> show d
  pure ()

readNetworkFile ::
  forall ls m.
  (MonadIO m, KnownNetwork 784 ls 10) =>
  FilePath ->
  m (NeuralNetwork 784 ls 10 Double)
readNetworkFile inFile =
  either error pure . Persist.decode @(MNISTNN ls)
    =<< liftIO (BS.readFile inFile)

imageSize :: Int
imageSize = 28 * 28

pixelSize :: Int
pixelSize = 28

type PixelSize = 28

type ImageSize = PixelSize * PixelSize

type MNISTNN ls = NeuralNetwork ImageSize ls 10 Double

type BatchedNet =
  '[ L 'Lin 300
   , L 'BN 300
   , L (Act 'ReLU) 300
   , L 'Lin 50
   , L 'BN 50
   , L (Act 'ReLU) 50
   , L 'Aff 10
   , L (Act 'Softmax) 10
   ]

type PlainNet =
  '[ L 'Aff 300
   , L (Act 'ReLU) 300
   , L 'Aff 50
   , L (Act 'ReLU) 50
   , L 'Aff 10
   , L (Act 'Softmax) 10
   ]

adams :: AdamParams Double
adams = AdamParams {beta1 = 0.9, beta2 = 0.999, epsilon = 1e-16}

doTrain ::
  (MonadIO m, M.MonadUnliftIO m, RandomGenM g r (ResourceT m), PrimMonad m) =>
  g ->
  TrainOpts ->
  m ()
doTrain g TrainOpts {..} = do
  puts "* Training Mode"
  now <- liftIO getZonedTime
  createDirectoryIfMissing True modelDir
  let stamp = formatTime defaultTimeLocale "%Y%m%d-%H%M%S" now
  there <- doesFileExist modelFile
  batchedNN <-
    if there
      then do
        puts "# Model file found. Resuming training..."
        copyFile modelFile (modelFile <> "." <> stamp <> ".bak")
        readNetworkFile modelFile
      else do
        puts "# No model found. Initialising with seed..."
        randomNetwork globalStdGen batchedNetSeed

  runResourceT $ do
    (numTrains, trainDataSet) <- readMNISTDataDir trainDataDir
    let (numBat0, r) = numTrains `quotRem` batchSize
        numBatches
          | r == 0 = numBat0
          | otherwise = numBat0 + 1

    puts $
      printf
        "# %d training data given, divided into %d minibatches, each of size %d"
        numTrains
        numBatches
        batchSize
    let shuffWindow = batchSize * max 1 (numBatches `quot` 100)

        trainBatches =
          trainDataSet
            & shuffleBuffered g shuffWindow
            & S.map (\(a, d) -> ((a, toDigitVector d), d))
            & chunksOfVector batchSize
            & S.cycle
    (numTests, tests) <- readMNISTDataDir testDataDir
    let testDataSet = S.cycle tests
    puts $ printf "# %d test datasets given" numTests
    testAcc :!: testData' <- calcTestAccuracy numTests batchSize batchedNN testDataSet
    puts "---"
    puts $ printf "Initial Test Accuracy: %f%%" $ testAcc * 100
    let intvl = fromMaybe 1 outputInterval
        (blk, resid) = epochs `quotRem` intvl
        epcs =
          foldr (:) (replicate (min 1 resid) resid) $ replicate blk intvl
    ((net' :!: _) :!: _) <-
      foldlM
        (step numBatches numTests)
        ((batchedNN :!: 0) :!: (trainBatches :!: testData'))
        epcs
    liftIO $ BS.writeFile modelFile $ Persist.encode net'
    puts $ printf "Model file written to: %s" modelFile
  where
    modelFile = modelDir </> batchedName
    params =
      MNISTParams
        { timeStep = gamma
        , dumpingFactor = 0.01
        , adamParams = adams
        }

    step numBatches numTests ((net :!: !n) :!: (!batches :!: !tests)) epoch = do
      let n' = epoch + n
      puts $ printf "** Batch(es) %d..%d started." n n'
      net' :> rest <-
        S.fold (flip (train @28 params)) net id $
          S.map (fst . U.unzip) $
            S.splitAt (epoch * numBatches) batches
      testAcc :!: testData' <- calcTestAccuracy numTests batchSize net' tests
      puts $ printf "Test Accuracy: %f%%" $ testAcc * 100
      mask_ $
        liftIO (BS.writeFile modelFile $ Persist.encode net')
          `finally` puts (printf "Model file written to: %s" modelFile)
      pure ((net' :!: n') :!: (rest :!: testData'))

chunksOfVector :: (U.Unbox a, PrimMonad m) => Int -> Stream (Of a) m r -> Stream (Of (U.Vector a)) m r
chunksOfVector n =
  SS.chunksOf n
    >>> SS.mapped
      (L.impurely S.foldM (L.vectorM @_ @U.Vector))

calcTestAccuracy ::
  PrimMonad m =>
  Int ->
  Int ->
  NeuralNetwork 784 BatchedNet 10 Double ->
  Stream (Of (MNISTInput PixelSize, Digit)) (ResourceT m) r ->
  ResourceT m (Double :!: Stream (Of (MNISTInput PixelSize, Digit)) (ResourceT m) r)
calcTestAccuracy numBatch n net =
  S.splitAt numBatch
    >>> chunksOfVector n
    >>> S.map
      ( \dataSet ->
          let (inps, outs) = U.unzip dataSet
              ys' = predicts net inps
           in accuracy outs ys'
      )
    >>> L.purely S.fold L.mean
    >>> fmap tupleOf

tupleOf :: Of a b -> Pair a b
tupleOf (a :> b) = a :!: b

puts :: MonadIO m => String -> m ()
puts = liftIO . putStrLn

readMNISTDataDir ::
  MonadResource m =>
  FilePath ->
  m (Int, Stream (Of (MNISTInput PixelSize, Digit)) m ())
readMNISTDataDir dir = do
  (LabelFileHeader {..}, labels) <- parseLabelFileS $ Q.readFile (dir </> "labels.mnist")
  (ImageFileHeader {..}, imgs) <-
    parseImageFileS $ Q.readFile (dir </> "images.mnist")
  when (labelCount /= imageCount) $
    error $
      printf
        "%s: Given dataset has different numbers of labels and images : %d %d"
        dir
        labelCount
        imageCount
  when
    ( fromIntegral columnCount /= pixelSize
        || fromIntegral rowCount /= pixelSize
    )
    $ error $
      printf
        "%s: Input image must be 28x28, but got %dx%d"
        dir
        rowCount
        columnCount
  let imgs' = S.map (toMNISTInput . unsafeToMat) imgs
  pure (fromIntegral labelCount, S.zip imgs' labels)

batchedNetSeed :: Network LayerSpec ImageSize BatchedNet 10 Double
batchedNetSeed =
  linear @300 @ImageSize (sqrt $ 2.0 / fromIntegral imageSize)
    :- batchnorm
    :- reLU_
    :- linear @50 (sqrt $ 2.0 / 300)
    :- batchnorm
    :- reLU_
    :- affine @10 (sqrt $ 1.0 / 50)
    :- softmax_
    :- Output

plainNetSeed :: Network LayerSpec ImageSize PlainNet 10 Double
plainNetSeed =
  affine @300 (sqrt $ 2.0 / fromIntegral imageSize)
    :- reLU_
    :- affine @50 (sqrt $ 2.0 / 300.0)
    :- reLU_
    :- affine @10 (sqrt $ 1.0 / 50)
    :- softmax_
    :- Output

data Net = BatchNormed | Plain
  deriving (Show, Eq, Ord, Generic)

data Cmd
  = Train TrainOpts
  | Recognise RecognitionOpts
  deriving (Show, Eq, Ord, Generic)

cmdP :: Opts.ParserInfo Cmd
cmdP =
  Opts.info (p <**> Opts.helper) $
    mconcat
      [ Opts.header "circles - hidden layer demo (Day 2)"
      , Opts.progDesc "Binary point classification with hidden layers"
      ]
  where
    p =
      Opts.subparser $
        mconcat
          [ Opts.command "train" $
              Opts.info (Train <$> trainOptsP) (Opts.progDesc "Train network for digit recognition")
          , Opts.command "recognise" $
              Opts.info (Recognise <$> recogniseOptsP) (Opts.progDesc "Run network to recognise hand-written digit.")
          ]

data TrainOpts = TrainOpts
  { epochs :: !Int
  , batchSize :: !Int
  , outputInterval :: !(Maybe Int)
  , gamma :: !Double
  , modelDir :: !FilePath
  , trainDataDir :: !FilePath
  , testDataDir :: !FilePath
  }
  deriving (Show, Eq, Ord)

trainOptsP :: Opts.Parser TrainOpts
trainOptsP = do
  epochs <-
    Opts.option Opts.auto $
      Opts.short 'n' <> Opts.long "epochs"
        <> Opts.value 10
        <> Opts.showDefault
        <> Opts.metavar "N"
        <> Opts.help "# of epochs"
  gamma <-
    Opts.option Opts.auto $
      Opts.long "gamma"
        <> Opts.short 'g'
        <> Opts.value 0.001
        <> Opts.metavar "GAMMA"
        <> Opts.showDefault
        <> Opts.help "Learning rate"
  outputInterval <-
    optional $
      Opts.option Opts.auto $
        Opts.long "interval" <> Opts.short 'I'
          <> Opts.help "Output interval"
  batchSize <-
    Opts.option Opts.auto $
      Opts.long "batch"
        <> Opts.short 'b'
        <> Opts.value 100
        <> Opts.showDefault
        <> Opts.help "Mini batch size"
  modelDir <-
    Opts.strOption $
      Opts.long "models" <> Opts.short 'M'
        <> Opts.metavar "DIR"
        <> Opts.value (workDir </> "models")
        <> Opts.showDefault
        <> Opts.help "The directory to save the trained model(s)."
  trainDataDir <-
    Opts.strOption $
      Opts.long "train-set" <> Opts.metavar "DIR"
        <> Opts.help "The directory containing training dataset; it must have images.mnist and labels.mnist"
        <> Opts.value ("data" </> "mnist" </> "train")
  testDataDir <-
    Opts.strOption $
      Opts.long "test-set" <> Opts.metavar "DIR"
        <> Opts.help "The directory containing test dataset; it must have images.mnist and labels.mnist"
        <> Opts.value ("data" </> "mnist" </> "test")
  pure TrainOpts {..}

data RecognitionOpts = RecognitionOpts {modelDir :: !FilePath, input :: !FilePath}
  deriving (Show, Eq, Ord, Generic)

recogniseOptsP :: Opts.Parser RecognitionOpts
recogniseOptsP = do
  modelDir <-
    Opts.strOption $
      Opts.short 'M' <> Opts.long "models" <> Opts.metavar "DIR"
        <> Opts.value (workDir </> "models")
        <> Opts.help "The directory containing model(s)"
  input <-
    Opts.strArgument $
      Opts.metavar "FILE"
        <> Opts.help "A path to the gray-scale image to recognise"
  pure RecognitionOpts {..}

workDir :: FilePath
workDir = "workspace" </> "mnist"

putNetworkInfo :: KnownNat i => NeuralNetwork i hs o a -> IO ()
putNetworkInfo net =
  let NetworkStat {..} = networkStat net
      !lays = DL.toList layers
   in putStrLn $
        printf
          "** Network of %d layers (%s), %d parameters."
          (length lays)
          (show lays)
          (getSum parameters)
