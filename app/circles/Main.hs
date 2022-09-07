{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingVia #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module Main (main) where

import Control.Applicative (optional, (<**>))
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Lens hiding (Snoc)
import Control.Monad ((<=<))
import qualified Data.DList as DL
import Data.Foldable (foldlM, forM_)
import Data.Functor (void)
import qualified Data.List as List
import Data.List.NonEmpty (NonEmpty (..))
import qualified Data.List.NonEmpty as NE
import Data.List.Split (splitOn)
import Data.Maybe (fromMaybe)
import Data.Monoid (Sum (..))
import Data.Strict (Pair (..))
import Data.Time (defaultTimeLocale, formatTime, getZonedTime)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Backend.Rasterific
import Diagrams.Prelude (Diagram, alignB, alignT, bg, black, blend, centerXY, fc, green, lc, mkHeight, orange, p2, pad, scale, strokeOpacity, white, (===), (|||))
import qualified Diagrams.Prelude as Dia
import GHC.TypeNats
import Generic.Data
import Linear
import Linear.Affine
import Numeric.Natural (Natural)
import qualified Options.Applicative as Opts
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.IO (BufferMode (LineBuffering), hSetBuffering, stdout)
import System.Random.Stateful (globalStdGen)
import Text.Printf (printf)
import Text.Read (readMaybe)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  cmd <- Opts.execParser cmdP
  case cmd of
    Circles opts -> dualCircleTest opts
    Spirals opts -> dualSpiralTest opts

data Cmd = Circles Opts | Spirals Opts
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
          [ Opts.command "circles" $
              Opts.info (Circles <$> optsP) (Opts.progDesc "Classifies dual circle datasets")
          , Opts.command "spirals" $
              Opts.info (Spirals <$> optsP) (Opts.progDesc "Classifies dual spiral datasets")
          ]

data Opts = Opts
  { epochs :: !Int
  , gamma :: !Double
  , layers :: NonEmpty (NonEmpty Natural)
  }
  deriving (Show, Eq, Ord)

optsP :: Opts.Parser Opts
optsP = do
  epochs <-
    Opts.option Opts.auto $
      Opts.short 'n'
        <> Opts.value 500
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
  layers <-
    fmap (fromMaybe $ (128 :| []) :| []) $
      optional $
        NE.some1 $
          Opts.option (Opts.maybeReader intsP) $
            Opts.long "layers"
              <> Opts.short 'L'
  pure Opts {..}

intsP :: String -> Maybe (NonEmpty Natural)
intsP = NE.nonEmpty <=< mapM readMaybe . splitOn ","

workDir, circleWorkDir, spiralWorkDir :: FilePath
workDir = "workspace"
circleWorkDir = workDir </> "circles"
spiralWorkDir = workDir </> "spirals"

savePointImage :: FilePath -> U.Vector ClusteredPoint -> IO ()
savePointImage fp pts =
  renderRasterific fp (mkHeight 256) $
    drawClusteredPoints pts & bg white

mkPredictionImage ::
  (Dia.N b ~ Double, Dia.V b ~ V2, Dia.Renderable (Dia.Path V2 Double) b) =>
  NeuralNetwork 2 ls 1 Double ->
  Vector ClusteredPoint ->
  Diagram b
mkPredictionImage nn pts =
  mconcat
    [ drawClusteredPoints pts & lc black & strokeOpacity 1.0
    , pixelateCluster
        64
        (\α -> blend (min 1.0 $ max 0.0 α) green orange)
        (p2 (-1.25, -1.25))
        (p2 (1.25, 1.25))
        nn
    ]

savePredictionComparisonImage ::
  FilePath ->
  NeuralNetwork 2 ls 1 Double ->
  (String, Vector ClusteredPoint) ->
  (String, Vector ClusteredPoint) ->
  IO ()
savePredictionComparisonImage fp nn (lab0, pts0) (lab1, pts1) =
  renderRasterific fp (mkHeight 256) $
    ( ( (mkPredictionImage nn pts0 & centerXY & alignB)
          === (texterific lab0 & scale 0.2 & fc white & centerXY & pad 1.5 & alignT)
      )
        ||| ( (mkPredictionImage nn pts1 & centerXY & alignB)
                === (texterific lab1 & scale 0.2 & fc white & centerXY & pad 1.1 & alignT)
            )
    )
      & centerXY
      & pad 1.1
      & bg green

showDim :: Show a => [a] -> String
showDim = List.intercalate "x" . map show

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

dualCircleTest :: Opts -> IO ()
dualCircleTest Opts {..} = do
  createDirectoryIfMissing True circleWorkDir
  trainSet <- evaluate . force =<< dualCircles globalStdGen 200 0.6 0.1
  testSet <- evaluate . force =<< dualCircles globalStdGen 100 0.6 0.1

  savePointImage (circleWorkDir </> "train.png") trainSet
  savePointImage (circleWorkDir </> "test.png") testSet

  putStrLn $ replicate 20 '-'
  putStrLn $
    printf "* Circle isolation, Circle isolation, %d epochs, learning rate = %f" epochs gamma

  forM_ layers $ \lay -> withSimpleNetwork (map (,ReLU) $ NE.toList lay) $ \seeds -> do
    !net0 <- randomNetwork globalStdGen seeds
    putNetworkInfo net0

    putStrLn $ printf "Initial training accuracy: %f%%" $ predictionAccuracy net0 trainSet * 100
    putStrLn $ printf "Initial validation accuracy: %f%%" $ predictionAccuracy net0 testSet * 100
    !net' <- evaluate $ trainByGradientDescent gamma epochs trainSet net0

    savePredictionComparisonImage
      ( circleWorkDir
          </> printf "predict-gd-%s.png" (showDim $ NE.toList lay)
      )
      net'
      ("Train", trainSet)
      ("Test", testSet)

    putStrLn $ printf "Training accuracy (GD): %f" $ predictionAccuracy net' trainSet * 100
    putStrLn $ printf "Validation accuracy (GD): %f" $ predictionAccuracy net' testSet * 100

adams :: AdamParams Double
adams = AdamParams {beta1 = 0.9, beta2 = 0.999, epsilon = 1e-16}

dualSpiralTest :: Opts -> IO ()
dualSpiralTest Opts {..} = do
  now <- getZonedTime
  let stamp = formatTime defaultTimeLocale "%Y%m%d-%H%M%S" now
      workDir = spiralWorkDir </> stamp
  createDirectoryIfMissing True workDir
  trainSet <- evaluate . force =<< dualSpirals globalStdGen 400 0.05
  testSet <- evaluate . force =<< dualSpirals globalStdGen 200 0.05
  putStrLn ""
  putStrLn $ replicate 20 '-'
  putStrLn $
    printf
      "* Dual spiral classification, %d epochs, learn rate = %f"
      epochs
      gamma
  savePointImage (workDir </> "train.png") trainSet
  savePointImage (workDir </> "test.png") testSet

  forM_ layers $ \lay ->
    withSimpleNetwork (map (,ReLU) $ NE.toList lay) $ \seeds -> do
      let dimStr = showDim $ NE.toList lay
          layDir = workDir </> dimStr
      createDirectoryIfMissing True layDir
      !net0 <- evaluate =<< randomNetwork globalStdGen seeds
      putNetworkInfo net0

      let (qs, r) = epochs `quotRem` 10
          es
            | qs <= 0 = [epochs]
            | r == 0 = replicate 10 qs
            | otherwise = replicate 10 qs ++ [r]
      putStrLn $
        printf "Initial: training accuracy: %f%%"
          $! predictionAccuracy net0 trainSet * 100
      putStrLn $
        printf "Initial: Validation accuracy: %f%%"
          $! predictionAccuracy net0 testSet * 100

      savePredictionComparisonImage
        (layDir </> "initial.png")
        net0
        ("Train", trainSet)
        ("Test", testSet)

      void $
        foldlM
          ( \(total :!: (netGD0 :!: netAdam0)) n -> do
              let !total' = total + n
              putStrLn $ printf "*** Epoch %d" total'
              !netGD <- evaluate $ trainByGradientDescent gamma n trainSet netGD0
              putStrLn $
                printf "[Gradient Descent] Training accuracy: %f%%"
                  $! predictionAccuracy netGD trainSet * 100
              putStrLn $
                printf "[Gradient Descent] Validation accuracy: %f%%"
                  $! predictionAccuracy netGD testSet * 100
              savePredictionComparisonImage
                (layDir </> printf "predict-gd-%d.png" total')
                netGD
                ("Train", trainSet)
                ("Test", testSet)
              putStrLn "---"
              !netAdam <- evaluate $ trainByAdam gamma adams n trainSet netAdam0
              putStrLn $
                printf "[Adam] Training accuracy: %f%%"
                  $! predictionAccuracy netAdam trainSet * 100
              putStrLn $
                printf "[Adam] Validation accuracy: %f%%"
                  $! predictionAccuracy netAdam testSet * 100
              savePredictionComparisonImage
                (layDir </> printf "predict-adam-%d.png" total')
                netAdam
                ("Train", trainSet)
                ("Test", testSet)

              pure $ total' :!: (netGD :!: netAdam)
          )
          (0 :!: (net0 :!: net0))
          es