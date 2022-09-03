{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE RecordWildCards #-}

module DeepLearning.Circles
  ( module DeepLearning.Circles.Types,
    trainByGradientDescent,
    predict,
    predictionAccuracy,
  )
where

import Control.Arrow ((>>>))
import qualified Control.Foldl as L
import Control.Lens (re, view, (^.))
import Data.Coerce (coerce)
import Data.Vector.Generic.Lens (vectorTraverse)
import qualified Data.Vector.Unboxed as U
import DeepLearning.Circles.Types
import DeepLearning.NeuralNetowrk.HigherKinded
import Linear (V1, V2)
import Linear.Affine

trainByGradientDescent ::
  Double ->
  Int ->
  U.Vector ClusteredPoint ->
  NeuralNetwork V2 hs V1 Double ->
  NeuralNetwork V2 hs V1 Double
trainByGradientDescent gamma epochs =
  trainGD gamma epochs crossEntropy
    . U.map \(ClusteredPoint pt clus) ->
      (pt ^. _Point, realToFrac $ fromEnum clus)

predict :: NeuralNetwork V2 hs V1 Double -> Point V2 Double -> Cluster
predict net =
  view _Point >>> evalNN net >>> \p -> if p < 0.5 then Cluster0 else Cluster1

predictionAccuracy ::
  NeuralNetwork V2 hs V1 Double -> U.Vector ClusteredPoint -> Double
predictionAccuracy nn =
  L.foldOver vectorTraverse $
    L.premap
      ( \ClusteredPoint {..} ->
          if predict nn coord == cluster then 1.0 else 0.0
      )
      L.mean
