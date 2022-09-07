{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module DeepLearning.Circles
  ( module DeepLearning.Circles.Types,
    trainByGradientDescent,
    trainByAdam,
    AdamParams (..),
    predict,
    predictionAccuracy,
    pixelateCluster,
  )
where

import Control.Arrow ((>>>))
import Control.Lens (view, (^.))
import Control.Subcategory (czipWith)
import Control.Subcategory.Linear (SomeBatch (..), UMat, VectorSpace (sumS), computeM, dimVal, fromBatchData, generateMat, rowAt, splitRowAt, unMat, unVec)
import Data.Function ((&))
import Data.Functor.Product (Product (..))
import Data.Massiv.Array (Ix2 (..))
import qualified Data.Massiv.Array as M
import Data.Proxy (Proxy)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles.Types
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Prelude (Colour, Diagram, N, alignBL, fc, lc, moveTo, opacity, rect, square, strokeOpacity)
import qualified Diagrams.Prelude as Dia
import GHC.TypeNats (SomeNat (..), someNatVal)
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
  trainGDF gamma epochs crossEntropy
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
  trainAdamF gamma params epochs crossEntropy
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
predictionAccuracy = withKnownNetwork $ \nn dats ->
  let adjusted = U.map (\ClusteredPoint {..} -> Pair coord $ clusterNum cluster) dats
   in case fromBatchData adjusted of
        MkSomeBatch (inOuts :: UMat m 3 a) ->
          let (ins, outs) = splitRowAt @2 inOuts
              outs' = evalBatchNN nn $ computeM ins
              n = fromIntegral $ dimVal @m
           in sumS
                ( czipWith
                    (\l r -> if decodeCluster l == decodeCluster r then 1.0 else 0.0)
                    (computeM outs)
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
pixelateCluster divs toColour ll ur = withKnownNetwork $ \nn ->
  let V2 w h = ur .-. ll
      dx = min w h / fromIntegral divs
      xDiv = ceiling (w / dx)
      yDiv = ceiling (h / dx)
   in case someNatVal $ xDiv * yDiv of
        SomeNat (_ :: Proxy nPts) ->
          let pts = generateMat @nPts @2 $ \(i :. j) ->
                let (xN, yN) = i `quotRem` fromIntegral xDiv
                    pt = ll .+^ dx *^ V2 (fromIntegral xN) (fromIntegral yN)
                 in if j == 0 then pt ^. _x else pt ^. _y
              !preds = rowAt @0 $ evalBatchNN nn pts
              !mainDia =
                M.foldMono
                  id
                  $ M.zipWith
                    ( \cds f ->
                        let col = toColour f
                            pt = P $ V2 (cds M.! 0) (cds M.! 1)
                         in square dx
                              & alignBL
                              & lc col
                              & fc col
                              & moveTo pt
                    )
                    (M.outerSlices $ unMat pts)
                    (unVec preds)
           in mainDia
                <> (rect w h & alignBL & moveTo ll & opacity 0.0 & strokeOpacity 0.0)
