{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module Main (main) where

import Control.Applicative ((<**>))
import Control.DeepSeq (force)
import Control.Exception (evaluate)
import DeepLearning.Circles
import DeepLearning.NeuralNetowrk.HigherKinded
import Linear.V (V)
import qualified Options.Applicative as Opts
import System.Random
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

main :: IO ()
main = do
  trainSet <- evaluate . force =<< dualCircles globalStdGen 200
  testSet <- evaluate . force =<< dualCircles globalStdGen 100
  Opts {..} <- Opts.execParser optsP
  putStrLn $ replicate 20 '-'
  putStrLn "* Circle isolation"

  putStrLn $
    printf "** 1 Hidden layer of 128 neurons, %d epochs, gamma = %f" iteration learningRate

  net128 <-
    generateNetworkA (randomRIO (0 :: Double, 1.0)) $
      reLUA @(V 128) :- sigmoidA :- Output

  putStrLn $ printf "Initial training accuracy (GD): %.5f%%" $ predictionAccuracy net128 trainSet * 100
  putStrLn $ printf "Initial validation accuracy (GD): %.5f%%" $ predictionAccuracy net128 testSet * 100
  let net' = trainByGradientDescent learningRate iteration testSet net128
  putStrLn $ printf "Training accuracy (GD): %.5f%%" $ predictionAccuracy net' trainSet * 100
  putStrLn $ printf "Validation accuracy (GD): %.5f%%" $ predictionAccuracy net' testSet * 100

  pure undefined
