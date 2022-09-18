{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module DeepLearning.Circles (
  module DeepLearning.Circles.Types,
  trainByGradientDescent,
  trainByAdam,
  AdamParams (..),
  predict,
  predictionAccuracy,
  pixelateCluster,
) where

import Control.Arrow ((>>>))
import Control.Lens (view, (^.))
import Control.Subcategory (czipWith)
import Control.Subcategory.Linear (SomeBatch (..), UMat, VectorSpace (sumS), dimVal, fromBatchData, rowAt, unVec, withDataPair)
import Data.Function ((&))
import qualified Data.Massiv.Array as M
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles.Types
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Prelude (Colour, Diagram, N, alignBL, fc, lc, moveTo, opacity, rect, square, strokeOpacity)
import qualified Diagrams.Prelude as Dia
import Linear
import Linear.Affine
import Type.Reflection (Typeable)

trainByGradientDescent ::
  Double ->
  Int ->
  U.Vector ClusteredPoint ->
  NeuralNetwork 2 hs 1 Double ->
  NeuralNetwork 2 hs 1 Double
trainByGradientDescent gamma epochs =
  trainGDF gamma 0.1 epochs logLikelihood
    . U.map \(ClusteredPoint pt clus) -> (pt ^. _Point, clusterNum clus)

clusterNum :: Cluster -> V1 Double
clusterNum = realToFrac . fromEnum

trainByAdam ::
  Double ->
  AdamParams Double ->
  Int ->
  U.Vector ClusteredPoint ->
  NeuralNetwork 2 hs 1 Double ->
  NeuralNetwork 2 hs 1 Double
trainByAdam gamma params epochs =
  trainAdamF gamma 0.1 params epochs logLikelihood
    . U.map \(ClusteredPoint pt clus) -> (pt ^. _Point, clusterNum clus)

predict :: (V2 Double -> V1 Double) -> Point V2 Double -> Cluster
predict f = view _Point >>> f >>> decodeCluster . view _x

decodeCluster :: Double -> Cluster
decodeCluster p =
  if not $ isNaN p && isInfinite p
    then if p < 0.5 then Cluster0 else Cluster1
    else error "Nan!"

predictionAccuracy ::
  NeuralNetwork 2 ls 1 Double -> U.Vector ClusteredPoint -> Double
predictionAccuracy = withKnownNeuralNetwork $ \nn dats ->
  let adjusted = U.map (\ClusteredPoint {..} -> (coord, clusterNum cluster)) dats
   in withDataPair adjusted $ \(ins :: UMat m i a, outs) ->
        let outs' = evalBatchNN nn ins
            n = fromIntegral $ dimVal @m
         in sumS
              ( czipWith
                  (\l r -> if decodeCluster l == decodeCluster r then 1.0 else 0.0)
                  outs
                  outs'
              )
              / n

pixelateCluster ::
  (N b ~ Double, Dia.V b ~ V2, Typeable Double, Dia.Renderable (Dia.Path V2 Double) b) =>
  Int ->
  (Double -> Colour Double) ->
  -- | Lower left point
  Point V2 Double ->
  -- | Upper right point
  Point V2 Double ->
  NeuralNetwork 2 ls 1 Double ->
  Diagram b
pixelateCluster divs toColour ll ur = withKnownNeuralNetwork $ \nn ->
  let V2 w h = ur .-. ll
      dx = min w h / fromIntegral divs
      xDiv = ceiling (w / dx)
      yDiv = ceiling (h / dx)
      pts0 =
        U.fromList
          [ ll .+^ dx *^ V2 (fromIntegral xN) (fromIntegral yN)
          | xN <- [0 .. xDiv - 1 :: Int]
          , yN <- [0 .. yDiv - 1 :: Int]
          ]
   in case fromBatchData pts0 of
        MkSomeBatch pts ->
          let !preds = rowAt @0 $ evalBatchNN nn pts
              !mainDia =
                M.foldMono
                  id
                  $ M.zipWith
                    ( \pt f ->
                        let col = toColour f
                         in square dx & alignBL & lc col & fc col & moveTo pt
                    )
                    (M.fromUnboxedVector M.Par pts0)
                    (unVec preds)
           in mainDia
                <> (rect w h & alignBL & moveTo ll & opacity 0.0 & strokeOpacity 0.0)
