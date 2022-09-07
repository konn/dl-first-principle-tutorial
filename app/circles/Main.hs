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
import Data.Foldable (foldlM, forM_)
import Data.Functor (void)
import qualified Data.List as List
import Data.List.NonEmpty (NonEmpty (..))
import qualified Data.List.NonEmpty as NE
import Data.List.Split (splitOn)
import Data.Maybe (fromMaybe)
import Data.Strict (Pair (..))
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Backend.Rasterific
import Diagrams.Prelude (Diagram, alignB, alignT, bg, black, blend, centerXY, fc, green, lc, mkHeight, orange, p2, pad, scale, strokeOpacity, white, (===), (|||))
import qualified Diagrams.Prelude as Dia
import Generic.Data
import Linear
import Linear.Affine
import Numeric.Natural (Natural)
import qualified Options.Applicative as Opts
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random.Stateful (globalStdGen)
import Text.Printf (printf)
import Text.Read (readMaybe)

main :: IO ()
main = do
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
    , pixelateScalarField
        64
        (view (_x @V1) . evalF nn . view _Point)
        (\α -> blend (min 1.0 $ max 0.0 α) green orange)
        (p2 (-1.25, -1.25))
        (p2 (1.25, 1.25))
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
  let NetworkStat{..} = networkStat net
      !lays = DL.toList layers
  in putStrLn $ 
        printf "** Network of %d layers (%s), %d parameters."
        (length lays) (show lays) parameters

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
    net0 <- randomNetwork globalStdGen seeds
    putNetworkInfo net0

    putStrLn $ printf "Initial training accuracy (GD): %f" $ predictionAccuracy net0 trainSet * 100
    putStrLn $ printf "Initial validation accuracy (GD): %f" $ predictionAccuracy net0 testSet * 100
    let net' = trainByGradientDescent gamma epochs trainSet net0

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
  createDirectoryIfMissing True spiralWorkDir
  trainSet <- evaluate . force =<< dualSpirals globalStdGen 400 0.05
  testSet <- evaluate . force =<< dualSpirals globalStdGen 200 0.05
  putStrLn ""
  putStrLn $ replicate 20 '-'
  putStrLn $
    printf
      "* Dual spiral classification, %d epochs, learn rate = %f"
      epochs
      gamma
  savePointImage (spiralWorkDir </> "train.png") trainSet
  savePointImage (spiralWorkDir </> "test.png") testSet

  forM_ layers $ \lay -> withSimpleNetwork (map (,ReLU) $ NE.toList lay) $ \seeds -> do
    net0 <- randomNetwork globalStdGen seeds
    putNetworkInfo net0
    let (qs, r) = epochs `quotRem` 10
        es
          | qs <= 0 = [epochs]
          | r == 0 = replicate 10 qs
          | otherwise = replicate 10 qs ++ [r]
    void $
      foldlM
        ( \(total :!: net) n -> do
            let !total' = total + n
            !netGD <- evaluate $ trainByGradientDescent gamma n trainSet net
            putStrLn $
              printf "Epoch %d: training accuracy (GD): %f" total'
                $! predictionAccuracy net trainSet * 100
            putStrLn $
              printf "Epoch %d: Validation accuracy (GD): %f" total'
                $! predictionAccuracy netGD testSet * 100
            savePredictionComparisonImage
              ( spiralWorkDir
                  </> printf "predict-gd-%s-%d.png" (showDim $ NE.toList lay) total'
              )
              netGD
              ("Train", trainSet)
              ("Test", testSet)
            pure $ total' :!: netGD
        )
        (0 :!: net0)
        es
    void $
      foldlM
        ( \(total :!: net) n -> do
            let !total' = total + n
            !netAdam <- evaluate $ trainByAdam gamma adams n trainSet net
            putStrLn $
              printf "Epoch %d: training accuracy (Adam): %f" total'
                $! predictionAccuracy net trainSet * 100
            putStrLn $
              printf "Epoch %d: Validation accuracy (Adam): %f" total'
                $! predictionAccuracy netAdam testSet * 100
            savePredictionComparisonImage
              ( spiralWorkDir
                  </> printf "predict-adam-%s-%d.png" (showDim $ NE.toList lay) total'
              )
              netAdam
              ("Train", trainSet)
              ("Test", testSet)
            pure $ total' :!: netAdam
        )
        (0 :!: net0)
        es
