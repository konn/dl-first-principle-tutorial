{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Applicative ((<**>))
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import Control.Lens
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.HigherKinded
import Diagrams.Backend.Rasterific
import Diagrams.Prelude (bg, black, blend, blue, dims2D, green, lc, orange, p2, red, strokeOpacity, white)
import Linear
import Linear.Affine
import Linear.V (V)
import qualified Options.Applicative as Opts
import System.Directory (createDirectoryIfMissing)
import System.FilePath ((</>))
import System.Random.Stateful (globalStdGen)
import Text.Printf (printf)

data Opts = Opts {iteration :: !Int, learningRate :: !Double}
  deriving (Show, Eq, Ord)

optsP :: Opts.ParserInfo Opts
optsP =
  Opts.info (p <**> Opts.helper) $
    Opts.progDesc "A simple iris classifier based on a single layer neural network"
  where
    p = do
      iteration <-
        Opts.option Opts.auto $
          Opts.short 'n'
            <> Opts.value 500
            <> Opts.showDefault
            <> Opts.metavar "N"
            <> Opts.help "# of iteration"
      learningRate <-
        Opts.option Opts.auto $
          Opts.long "gamma"
            <> Opts.short 'g'
            <> Opts.value 0.001
            <> Opts.metavar "GAMMA"
            <> Opts.showDefault
            <> Opts.help "Learning rate"
      pure Opts {..}

workDir :: String
workDir = "workspace"

main :: IO ()
main = do
  trainSet <- evaluate . force =<< dualCircles globalStdGen 200 0.6 0.1
  testSet <- evaluate . force =<< dualCircles globalStdGen 100 0.6 0.1
  Opts {..} <- Opts.execParser optsP

  createDirectoryIfMissing True workDir
  renderRasterific (workDir </> "train.png") (dims2D 256 256) $
    drawClusteredPoints trainSet & bg white
  renderRasterific (workDir </> "test.png") (dims2D 256 256) $
    drawClusteredPoints testSet & bg white

  putStrLn $ replicate 20 '-'
  putStrLn "* Circle isolation"

  putStrLn $
    printf "** 1 Hidden layer of 128 neurons, %d epochs, gamma = %f" iteration learningRate

  net128 <-
    generateNetworkA $
      unitRandom globalStdGen ReLU
        :- unitRandom @(V 128) globalStdGen Sigmoid
        :- Output

  putStrLn $ printf "Initial training accuracy (GD): %f" $ predictionAccuracy net128 trainSet * 100
  putStrLn $ printf "Initial validation accuracy (GD): %f" $ predictionAccuracy net128 testSet * 100
  let net' = trainByGradientDescent learningRate iteration trainSet net128

  renderRasterific (workDir </> "test-predict.png") (dims2D 256 256) $
    mconcat
      [ drawClusteredPoints testSet & lc black & strokeOpacity 1.0
      , pixelateScalarField
          64
          (view _x . evalNN net' . view _Point)
          (\α -> blend (min 1.0 $ max 0.0 α) green orange)
          (p2 (-1.25, -1.25))
          (p2 (1.25, 1.25))
      ]

  putStrLn $ printf "Training accuracy (GD): %f" $ predictionAccuracy net' trainSet * 100
  putStrLn $ printf "Validation accuracy (GD): %f" $ predictionAccuracy net' testSet * 100
