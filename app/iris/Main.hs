{-# LANGUAGE BangPatterns #-}

module Main (main) where

import qualified Control.Foldl as L
import qualified Data.ByteString.Lazy as LBS
import Data.Format.SpaceSeparated
import Data.Functor.Compose (Compose (..))
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Iris
import Linear
import System.Random

main :: IO ()
main = do
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
  let loss :: IrisVector (IrisFeatures Double) -> (Double, Double)
      loss w =
        let !y = U.map (forward w) features
         in L.foldOver
              vectorTraverse
              ( (,)
                  <$> L.premap (quadrance . uncurry (^-^)) L.sum
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

  let !w100 = train 0.001 100 (U.zip features classes) w0
  putStrLn $ "(loss, accuracy) after 100 steps: " <> show (loss w100)
  let !w200 = train 0.001 100 (U.zip features classes) w100
  putStrLn $ "(loss, accuracy) after 200 steps: " <> show (loss w200)
  let !w500 = train 0.001 300 (U.zip features classes) w100
  putStrLn $ "(loss, accuracy) after 500 steps: " <> show (loss w500)
