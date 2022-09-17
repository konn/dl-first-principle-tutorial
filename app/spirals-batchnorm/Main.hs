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

import Control.Applicative ((<**>))
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Lens hiding (Snoc)
import qualified Data.DList as DL
import Data.Foldable (foldlM)
import Data.Functor (void)
import Data.Monoid (Sum (..))
import Data.Strict (Pair (..))
import Data.Time (defaultTimeLocale, formatTime, getZonedTime)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Backend.Rasterific
import Diagrams.Prelude (Diagram, alignB, alignT, bg, black, blend, centerXY, fc, green, lc, mkHeight, orange, p2, pad, strokeOpacity, white, (===), (|||))
import qualified Diagrams.Prelude as Dia
import GHC.TypeNats
import Linear
import Linear.Affine
import qualified Options.Applicative as Opts
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.IO (BufferMode (LineBuffering), hSetBuffering, stdout)
import System.Random.Stateful (globalStdGen)
import Text.Printf (printf)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  opts <- Opts.execParser gloptP
  dualSpiralTest opts

gloptP :: Opts.ParserInfo Opts
gloptP =
  Opts.info (optsP <**> Opts.helper) $
    mconcat
      [ Opts.header "circles - hidden layer demo (Day 2)"
      , Opts.progDesc "Binary point classification with hidden layers"
      ]

data Opts = Opts
  { epochs :: !Int
  , gamma :: !Double
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
  pure Opts {..}

workDir, spiralWorkDir :: FilePath
workDir = "workspace"
spiralWorkDir = workDir </> "spirals-batchnorm"

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
          === (texterific lab0 & Dia.scale 0.2 & fc white & centerXY & pad 1.5 & alignT)
      )
        ||| ( (mkPredictionImage nn pts1 & centerXY & alignB)
                === (texterific lab1 & Dia.scale 0.2 & fc white & centerXY & pad 1.1 & alignT)
            )
    )
      & centerXY
      & pad 1.1
      & bg green

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

adams :: AdamParams Double
adams = AdamParams {beta1 = 0.9, beta2 = 0.999, epsilon = 1e-16}

dualSpiralTest :: Opts -> IO ()
dualSpiralTest Opts {..} = do
  now <- getZonedTime
  let stamp = formatTime defaultTimeLocale "%Y%m%d-%H%M%S" now
      work = spiralWorkDir </> stamp
  createDirectoryIfMissing True work
  trainSet <- evaluate . force =<< dualSpirals globalStdGen 400 0.05
  testSet <- evaluate . force =<< dualSpirals globalStdGen 200 0.05
  putStrLn ""
  putStrLn $ replicate 20 '-'
  putStrLn $
    printf
      "* Dual spiral classification, %d epochs, learn rate = %f"
      epochs
      gamma
  savePointImage (work </> "train.png") trainSet
  savePointImage (work </> "test.png") testSet

  let seeds =
        linear @40 (2 / sqrt 2)
          :- batchnorm (sqrt $ 2 / 40)
          :- reLU_
          :- linear @25 (2 / 40)
          :- batchnorm (sqrt $ 2 / 25)
          :- reLU_
          :- affine @10 (2 / sqrt 25)
          :- reLU_
          :- affine (1 / sqrt 10)
          :- sigmoid_
          :- Output
      layDir = work
  createDirectoryIfMissing True layDir
  !net0 <- evaluate =<< randomNetwork globalStdGen seeds
  putNetworkInfo net0

  let (qs, r) = epochs `quotRem` 10
      es
        | qs <= 0 = [epochs]
        | r == 0 = replicate 10 qs
        | otherwise = replicate 10 qs ++ [r]
  putStrLn $
    printf "Initial: training accuracy: %f%%" $!
      predictionAccuracy net0 trainSet * 100
  putStrLn $
    printf "Initial: Validation accuracy: %f%%" $!
      predictionAccuracy net0 testSet * 100

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
            printf "[Gradient Descent] Training accuracy: %f%%" $!
              predictionAccuracy netGD trainSet * 100
          putStrLn $
            printf "[Gradient Descent] Validation accuracy: %f%%" $!
              predictionAccuracy netGD testSet * 100
          savePredictionComparisonImage
            (layDir </> printf "predict-gd-%d.png" total')
            netGD
            ("Train", trainSet)
            ("Test", testSet)
          putStrLn "---"
          !netAdam <- evaluate $ trainByAdam gamma adams n trainSet netAdam0
          putStrLn $
            printf "[Adam] Training accuracy: %f%%" $!
              predictionAccuracy netAdam trainSet * 100
          putStrLn $
            printf "[Adam] Validation accuracy: %f%%" $!
              predictionAccuracy netAdam testSet * 100
          savePredictionComparisonImage
            (layDir </> printf "predict-adam-%d.png" total')
            netAdam
            ("Train", trainSet)
            ("Test", testSet)

          pure $ total' :!: (netGD :!: netAdam)
      )
      (0 :!: (net0 :!: net0))
      es