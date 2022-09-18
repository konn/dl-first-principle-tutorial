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
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE NoStarIsType #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module Main (main) where

import Control.Applicative (optional, (<**>))
import Control.Arrow
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Foldl (EndoM (..))
import qualified Control.Foldl as L
import Control.Lens hiding (Snoc, (:>))
import Control.Monad (when)
import Control.Monad.IO.Class
import Control.Monad.Trans.Class (lift)
import Control.Monad.Trans.Resource (MonadResource (..))
import qualified Control.Monad.Trans.Resource as MR
import Control.Subcategory.Linear (UMat, unMat, unsafeToMat)
import qualified Data.Bifunctor as Bi
import qualified Data.ByteString as BS
import qualified Data.DList as DL
import Data.Foldable (fold, foldlM)
import Data.Functor (void)
import Data.Functor.Of (Of (..))
import qualified Data.Heap as H
import Data.Massiv.Array (PrimMonad, Sz (..))
import qualified Data.Massiv.Array as M
import Data.Massiv.Array.IO (readImageAuto)
import Data.Maybe (fromMaybe)
import Data.Monoid (Ap (..), Sum (..))
import qualified Data.Persist as Persist
import Data.Strict.Tuple (Pair (..))
import Data.Time (defaultTimeLocale, formatTime, getZonedTime)
import qualified Data.Vector.Unboxed as U
import Data.Word (Word32, Word8)
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
import System.Random.Stateful (FrozenGen (MutableGen, freezeGen, thawGen), RandomGenM, globalStdGen, uniformRM)
import Text.Printf (printf)
import UnliftIO (mask_)
import UnliftIO.Directory (copyFile, createDirectoryIfMissing, doesFileExist)
import qualified UnliftIO.Exception as UIO
import UnliftIO.Resource

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  Opts.execParser cmdP >>= \case
    Train opts -> do
      g <- freezeGen globalStdGen
      doTrain g opts
    Recognise opts -> recognise opts
    Resample opts -> do
      g <- freezeGen globalStdGen
      resample g opts

resample ::
  ( FrozenGen g IO
  , RandomGenM (MutableGen g IO) r (ResourceT IO)
  ) =>
  g ->
  ResampleOpts ->
  IO ()
resample seed ResampleOpts {..} = do
  g <- thawGen seed
  resampleWith g (dataDir </> "train") (dataDir </> "train_mini") trainDataSize
  resampleWith g (dataDir </> "test") (dataDir </> "test_mini") testDataSize

instance (MonadResource m, Functor f) => MonadResource (Stream f m) where
  liftResourceT = lift . MR.liftResourceT
  {-# INLINE liftResourceT #-}

resampleWith ::
  (M.MonadUnliftIO f, RandomGenM g r (ResourceT f)) =>
  g ->
  FilePath ->
  FilePath ->
  Word32 ->
  f ()
resampleWith g inDir outDir targetSize = do
  createDirectoryIfMissing True outDir
  runResourceT $ do
    (_, stream) <- readMNISTDataDir inDir
    let imgHeader =
          ImageFileHeader
            { imageCount = targetSize
            , rowCount = fromIntegral pixelSize
            , columnCount = fromIntegral pixelSize
            }
        lblHeader = LabelFileHeader {labelCount = targetSize}
    stream
      & L.impurely
        S.foldM_
        (resevoirSample g (fromIntegral targetSize) $ S.each <$> L.generalize L.list)
      & SS.effect
      & S.unzip
      & S.map unMat
      & streamImageFile imgHeader
      & Q.writeFile (outDir </> imagesFile)
      & streamLabelFile lblHeader
      & Q.writeFile (outDir </> labelsFile)

  pure ()

resevoirSample :: RandomGenM g r m => g -> Int -> L.FoldM m a b -> L.FoldM m a b
{-# INLINEABLE resevoirSample #-}
resevoirSample g n l =
  L.FoldM
    step
    (pure mempty)
    (L.foldM (L.premapM (pure . H.payload) l))
  where
    step resevoir a = do
      !w <- uniformRM (0.0 :: Double, 1) g
      let !entry = H.Entry {priority = w, payload = a}
          !res' = H.insert entry resevoir
      if H.size res' <= n
        then pure res'
        else pure $! H.deleteMin res'

data Cmd
  = Train TrainOpts
  | Recognise RecognitionOpts
  | Resample ResampleOpts
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
              Opts.info (Train <$> trainOptsP <**> Opts.helper) (Opts.progDesc "Train network for digit recognition")
          , Opts.command "recognise" $
              Opts.info (Recognise <$> recogniseOptsP <**> Opts.helper) (Opts.progDesc "Run network to recognise hand-written digit.")
          , Opts.command "resample" $
              Opts.info
                (Resample <$> resampleOptsP <**> Opts.helper)
                (Opts.progDesc "resmples train/test datasets")
          ]

resampleOptsP :: Opts.Parser ResampleOpts
resampleOptsP = do
  testDataSize <-
    Opts.option Opts.auto $
      Opts.long "test-size"
        <> Opts.value 10
        <> Opts.showDefault
        <> Opts.help "The size of test dataset after resampling"
  trainDataSize <-
    Opts.option Opts.auto $
      Opts.long "train-size"
        <> Opts.value 10
        <> Opts.showDefault
        <> Opts.help "The size of train dataset after resampling"
  dataDir <-
    Opts.strOption $
      Opts.short 'd'
        <> Opts.long "data-dir"
        <> Opts.metavar "DIR"
        <> Opts.value ("data" </> "mnist")
  pure ResampleOpts {..}

data ResampleOpts = ResampleOpts
  { testDataSize, trainDataSize :: !Word32
  , dataDir :: !FilePath
  }
  deriving (Show, Eq, Ord, Generic)

batchedName :: FilePath
batchedName = "batched.dat"

recognise :: RecognitionOpts -> IO ()
recognise RecognitionOpts {..} = do
  let modelPath = modelDir </> batchedName
  putStrLn $ "Recognising: " <> input
  batchedNet <- readNetworkFile @BatchedNet modelPath

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
  m (NeuralNetwork 784 ls 10 Float)
readNetworkFile inFile =
  either error pure . Persist.decode @(MNISTNN ls)
    =<< liftIO (BS.readFile inFile)

imageSize :: Int
imageSize = 28 * 28

pixelSize :: Int
pixelSize = 28

type PixelSize = 28

type ImageSize = PixelSize * PixelSize

type MNISTNN ls = NeuralNetwork ImageSize ls 10 Float

type BatchedNet =
  '[ L 'Lin 300
   , L 'BN 300
   , L (Act 'ReLU) 300
   , L 'Lin 50
   , L 'BN 50
   , L (Act 'ReLU) 50
   , L 'Lin 10
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

adams :: AdamParams Float
adams = AdamParams {beta1 = 0.9, beta2 = 0.999, epsilon = 1e-16}

doTrain ::
  forall m g r.
  ( MonadIO m
  , M.MonadUnliftIO m
  , PrimMonad m
  , FrozenGen g m
  , RandomGenM (MutableGen g m) r (ResourceT m)
  ) =>
  g ->
  TrainOpts ->
  m ()
doTrain seed TrainOpts {..} = do
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
        UIO.evaluate . force =<< readNetworkFile modelFile
      else do
        puts "# No model found. Initialising with seed..."
        UIO.evaluate . force =<< randomNetwork globalStdGen batchedNetSeed

  (numTrains, _) <-
    runResourceT $ readMNISTDataDir trainDataDir
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

      intvl = fromMaybe 1 outputInterval
      (blk, resid) = epochs `quotRem` intvl
      epcs =
        foldr (:) (replicate (min 1 resid) resid) $ replicate blk intvl

  void $ runResourceT $ do
    (numTests, tests0) <- readMNISTDataDir testDataDir
    puts $ printf "# %d test datasets given" numTests
    !testAcc <-
      calcTestAccuracy batchSize batchedNN $ S.map (Bi.first toMNISTInput) tests0
    puts $ printf "Initial Test Accuracy: %f%%" $ testAcc * 100
    pure numTests
  void $
    foldlM
      ( \(net :!: es) ecount -> do
          let !es' = es + ecount
          puts $ printf "** Epoch #%d..%d Started." es es'
          !net' <-
            appEndoM
              ( fold $
                  replicate
                    ecount
                    ( EndoM $ \ !nets -> do
                        g <- thawGen seed
                        runResourceT $ do
                          (_, !trainDataSet) <- readMNISTDataDir trainDataDir
                          let !trainBatches =
                                trainDataSet
                                  & S.map (Bi.first toMNISTInput)
                                  & shuffleBuffered g shuffWindow
                                  & S.map (\(a, d) -> ((a, toDigitVector d), d))
                                  & chunksOfVector batchSize
                          S.fold_ (flip (train @PixelSize params)) nets id $
                            S.map (fst . U.unzip) trainBatches
                    )
              )
              net
          runResourceT $ do
            (_, tests) <- readMNISTDataDir testDataDir
            !acc <-
              calcTestAccuracy batchSize net' $
                S.map (Bi.first toMNISTInput) tests
            puts $ printf "Test Accuracy: %f%%" $ acc * 100
          mask_ $ do
            liftIO $ BS.writeFile modelFile $ Persist.encode net'
            puts $ printf "Model file written to: %s" modelFile
          pure (net' :!: es')
      )
      (batchedNN :!: 0)
      epcs
  where
    modelFile = modelDir </> batchedName
    params =
      MNISTParams
        { timeStep = gamma
        , dumpingFactor
        , adamParams = adams
        }

chunksOfVector :: (U.Unbox a, PrimMonad m) => Int -> Stream (Of a) m r -> Stream (Of (U.Vector a)) m r
chunksOfVector n =
  SS.chunksOf n
    >>> SS.mapped
      (L.impurely S.foldM (L.vectorM @_ @U.Vector))

calcTestAccuracy ::
  PrimMonad m =>
  -- | Batchsize
  Int ->
  NeuralNetwork 784 BatchedNet 10 Float ->
  Stream (Of (MNISTInput PixelSize Float, Digit)) (ResourceT m) r ->
  ResourceT m Double
calcTestAccuracy n net =
  chunksOfVector n
    >>> S.map
      ( \dataSet ->
          let (inps, outs) = U.unzip dataSet
              ys' = predicts net inps
           in M.zipWith
                (\l r -> if l == r then 1.0 else 0.0)
                (M.fromUnboxedVector M.Par outs)
                (M.fromUnboxedVector M.Par ys')
      )
    >>> S.subst (getAp . M.foldMono (Ap . S.yield))
    >>> SS.concats
    >>> L.purely S.fold_ L.mean

puts :: MonadIO m => String -> m ()
puts = liftIO . putStrLn

readMNISTDataDir ::
  MonadResource m =>
  FilePath ->
  m (Int, Stream (Of (UMat PixelSize PixelSize Word8, Digit)) m ())
readMNISTDataDir dir = do
  (LabelFileHeader {..}, labels) <- parseLabelFileS $ Q.readFile (dir </> labelsFile)
  (ImageFileHeader {..}, imgs) <-
    parseImageFileS $ Q.readFile (dir </> imagesFile)
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
    $ error
    $ printf
      "%s: Input image must be 28x28, but got %dx%d"
      dir
      rowCount
      columnCount
  let imgs' = S.map unsafeToMat imgs
  pure (fromIntegral labelCount, S.zip imgs' labels)

imagesFile :: FilePath
imagesFile = "images.mnist"

labelsFile :: FilePath
labelsFile = "labels.mnist"

batchedNetSeed :: Network LayerSpec ImageSize BatchedNet 10 Float
batchedNetSeed =
  linear @300 @ImageSize 0.01
    :- batchnorm
    :- reLU_
    :- linear @50 0.01
    :- batchnorm
    :- reLU_
    :- linear @10 0.01
    :- softmax_
    :- Output

plainNetSeed :: Network LayerSpec ImageSize PlainNet 10 Float
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

data TrainOpts = TrainOpts
  { epochs :: !Int
  , batchSize :: !Int
  , outputInterval :: !(Maybe Int)
  , gamma :: !Float
  , dumpingFactor :: !Float
  , modelDir :: !FilePath
  , trainDataDir :: !FilePath
  , testDataDir :: !FilePath
  }
  deriving (Show, Eq, Ord)

trainOptsP :: Opts.Parser TrainOpts
trainOptsP = do
  epochs <-
    Opts.option Opts.auto $
      Opts.short 'n'
        <> Opts.long "epochs"
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
        Opts.long "interval"
          <> Opts.short 'I'
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
      Opts.long "models"
        <> Opts.short 'M'
        <> Opts.metavar "DIR"
        <> Opts.value (workDir </> "models")
        <> Opts.showDefault
        <> Opts.help "The directory to save the trained model(s)."
  trainDataDir <-
    Opts.strOption $
      Opts.long "train-set"
        <> Opts.metavar "DIR"
        <> Opts.help "The directory containing training dataset; it must have images.mnist and labels.mnist"
        <> Opts.value ("data" </> "mnist" </> "train")
  testDataDir <-
    Opts.strOption $
      Opts.long "test-set"
        <> Opts.metavar "DIR"
        <> Opts.help "The directory containing test dataset; it must have images.mnist and labels.mnist"
        <> Opts.value ("data" </> "mnist" </> "test")
  dumpingFactor <-
    Opts.option Opts.auto $
      Opts.long "alpha"
        <> Opts.long "dumping-factor"
        <> Opts.value 0.1
        <> Opts.help "dumping factor for moving average used in batchnorm layer"
        <> Opts.showDefault
  pure TrainOpts {..}

data RecognitionOpts = RecognitionOpts {modelDir :: !FilePath, input :: !FilePath}
  deriving (Show, Eq, Ord, Generic)

recogniseOptsP :: Opts.Parser RecognitionOpts
recogniseOptsP = do
  modelDir <-
    Opts.strOption $
      Opts.short 'M'
        <> Opts.long "models"
        <> Opts.metavar "DIR"
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
