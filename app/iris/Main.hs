{-# LANGUAGE ApplicativeDo #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE RecordWildCards #-}

module Main (main) where

import Control.Applicative ((<**>))
import qualified Control.Foldl as L
import qualified Data.ByteString.Lazy as LBS
import Data.Format.SpaceSeparated
import Data.Function (on)
import Data.Functor.Compose (Compose (..))
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Iris
import Linear
import qualified Options.Applicative as Opts
import System.Random

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
  opts@Opts {..} <- Opts.execParser optsP
  print opts
  features <-
    either error (pure . U.convert) . decodeSSV
      =<< LBS.readFile "data/iris/x.dat"
  classes <-
    either error (pure . U.convert) . decodeSSV
      =<< LBS.readFile "data/iris/y.dat"
  Compose !w0 <-
    sequenceA $
      pure $
        randomRIO (0, recip $ fromIntegral $ length $ U.head features)
  putStrLn $ "Initial matrix: " <> show w0
  let loss w =
        let !y = U.map (forward w) features
         in L.foldOver
              vectorTraverse
              ( (,)
                  <$> L.premap (quadrance . uncurry ((^-^) `on` normalize)) L.sum
                  <*> L.premap
                    ( \(l, r) ->
                        if classifyIris l == classifyIris r
                          then 1.0 :: Double
                          else 0.0
                    )
                    L.mean
              )
              $ U.zip classes y

  putStrLn $ "Initial (loss, accuracy): " <> show (loss w0)

  let step n = train learningRate n (U.zip features classes)

  let n = iteration `quot` 4
  if n > 0
    then do
      let !w1 = step n w0
      putStrLn $ "(loss, accuracy) after " <> show n <> " steps: " <> show (loss w1)
      let !w2 = step n w1
      putStrLn $ "(loss, accuracy) after " <> show (2 * n) <> " steps: " <> show (loss w2)
      let !w3 = step n w2
      putStrLn $ "(loss, accuracy) after " <> show (3 * n) <> " steps: " <> show (loss w3)
      let !w4 = step (iteration - n * 3) w3
      putStrLn $ "(loss, accuracy) after " <> show iteration <> " steps: " <> show (loss w4)
    else do
      let w' = step iteration w0
      putStrLn $ "(loss, accuracy) after " <> show iteration <> " steps: " <> show (loss w')
