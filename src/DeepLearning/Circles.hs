{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TypeApplications #-}

module DeepLearning.Circles
  ( module DeepLearning.Circles.Types,
    trainByGradientDescent,
    trainByAdam,
    AdamParams (..),
    predict,
    predictionAccuracy,
    pixelateScalarField,
  )
where

import Control.Arrow ((>>>))
import qualified Control.Foldl as L
import Control.Lens (view, (^.))
import Data.Function ((&))
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles.Types
import DeepLearning.NeuralNetowrk.Massiv
import Diagrams.Prelude (Colour, Diagram, N, alignBL, fc, lc, moveTo, opacity, rect, square, strokeOpacity)
import qualified Diagrams.Prelude as Dia
import Linear (V1 (..), V2 (..), (*^), (^/))
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
predict f =
  view _Point >>> f >>> \(V1 p) ->
    if not $ isNaN p && isInfinite p
      then if p < 0.5 then Cluster0 else Cluster1
      else error "Nan!"

predictionAccuracy ::
  (V2 Double -> V1 Double) -> U.Vector ClusteredPoint -> Double
predictionAccuracy f =
  L.foldOver vectorTraverse $
    L.premap
      ( \ClusteredPoint {..} ->
          if predict f coord == cluster then 1.0 else 0.0
      )
      L.mean

pixelateScalarField ::
  (N b ~ Double, Dia.V b ~ V2, Typeable Double, Dia.Renderable (Dia.Path V2 Double) b) =>
  Int ->
  (Point V2 Double -> Double) ->
  (Double -> Colour Double) ->
  -- | Lower left point
  Point V2 Double ->
  -- | Upper right point
  Point V2 Double ->
  Diagram b
pixelateScalarField divs field toColour ll ur =
  let V2 w h = ur .-. ll
      dx = min w h / fromIntegral divs
   in mconcat
        [ square dx
          & alignBL
          & lc col
          & fc col
          & moveTo pt
        | xN <- [0 :: Int .. ceiling (w / dx) - 1]
        , yN <- [0 :: Int .. ceiling (h / dx) - 1]
        , let pt = ll .+^ dx *^ V2 (fromIntegral xN) (fromIntegral yN)
              col = toColour $ field $ pt .+^ V2 dx dx ^/ 2.0
        ]
        <> (rect w h & alignBL & moveTo ll & opacity 0.0 & strokeOpacity 0.0)
