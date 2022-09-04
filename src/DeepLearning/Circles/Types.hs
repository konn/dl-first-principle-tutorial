{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DerivingStrategies #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE NoMonomorphismRestriction #-}
{-# OPTIONS_GHC -funbox-strict-fields #-}

module DeepLearning.Circles.Types
  ( dualCircles,
    dualSpirals,
    ClusteredPoint (..),
    Cluster (..),
    clusteredPointD,
    drawClusteredPoints,
  )
where

import Control.Arrow ((>>>))
import Control.Lens (foldMapOf, (^.))
import Control.Monad (guard, join)
import Data.Bit.ThreadSafe (Bit (..))
import Data.Coerce (coerce)
import Data.Csv (ToRecord (..))
import qualified Data.Csv as Csv
import Data.Function ((&))
import qualified Data.Vector as V
import Data.Vector.Generic.Lens
import qualified Data.Vector.Unboxed as U
import Data.Vector.Unboxed.Deriving (derivingUnbox)
import qualified Diagrams.Prelude as Dia
import GHC.Generics (Generic)
import Linear
import Linear.Affine
import System.Random.Stateful

data Cluster = Cluster0 | Cluster1
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)
  deriving anyclass (Uniform, Finite, Random)

c2b :: Cluster -> Bool
{-# INLINE c2b #-}
c2b = \case
  Cluster0 -> False
  Cluster1 -> True

b2c :: Bool -> Cluster
b2c = \case
  False -> Cluster0
  True -> Cluster1

instance UniformRange Cluster where
  uniformRM (l, r) g = do
    b2c <$> uniformRM (c2b l, c2b r) g

derivingUnbox
  "Cluster"
  [t|Cluster -> Bit|]
  [|coerce c2b|]
  [|coerce b2c|]

instance Csv.ToField Cluster where
  toField = Csv.toField . fromEnum
  {-# INLINE toField #-}

data ClusteredPoint = ClusteredPoint {coord :: !(Point V2 Double), cluster :: !Cluster}
  deriving (Show, Eq, Ord, Generic)

derivingUnbox
  "ClusteredPoint"
  [t|ClusteredPoint -> (Point V2 Double, Cluster)|]
  [|\ClusteredPoint {..} -> (coord, cluster)|]
  [|\(coord, cluster) -> ClusteredPoint {..}|]

instance Csv.DefaultOrdered ClusteredPoint where
  headerOrder = const $ V.fromList ["x", "y", "cluster"]
  {-# INLINE headerOrder #-}

instance Csv.FromField Cluster where
  parseField x = do
    i <- Csv.parseField x
    guard $ i == 0 || i == 1
    pure $ toEnum i
  {-# INLINE parseField #-}

instance Csv.ToNamedRecord ClusteredPoint where
  toNamedRecord ClusteredPoint {..} =
    Csv.namedRecord
      [ ("x", Csv.toField $ coord ^. _x)
      , ("y", Csv.toField $ coord ^. _y)
      , ("cluster", Csv.toField cluster)
      ]

instance ToRecord ClusteredPoint where
  toRecord ClusteredPoint {..} =
    Csv.record
      [ Csv.toField $ coord ^. _x
      , Csv.toField $ coord ^. _y
      , Csv.toField cluster
      ]

instance Csv.FromRecord ClusteredPoint where
  parseRecord v =
    ClusteredPoint
      <$> (fmap P . V2 <$> v Csv..! 0 <*> v Csv..! 1)
      <*> v Csv..! 2
  {-# INLINE parseRecord #-}

instance Csv.FromNamedRecord ClusteredPoint where
  parseNamedRecord r =
    ClusteredPoint
      <$> (fmap P . V2 <$> r Csv..: "x" <*> r Csv..: "y")
      <*> r Csv..: "cluster"

{- |
Randomly generates clusters of points scatterred roughly
around two distinct concentral circles.
-}
dualCircles :: RandomGenM g r m => g -> Int -> Double -> Double -> m (U.Vector ClusteredPoint)
dualCircles g n factor noise = U.replicateM n $ do
  ns <- sequence $ join V2 $ randomRM (0, noise) g
  cluster <- randomM g
  let r = case cluster of
        Cluster0 -> factor
        Cluster1 -> 1.0
  theta <- randomRM (-pi, pi) g
  let coord = P $ r *^ V2 (cos theta) (sin theta) ^+^ ns
  pure ClusteredPoint {..}

-- | Dual spirals
dualSpirals ::
  RandomGenM g r m =>
  g ->
  Int ->
  Double ->
  m (Vector ClusteredPoint)
dualSpirals g n noise = U.replicateM n $ do
  theta <- randomRM (0, maxTheta) g
  dev <- sequenceA $ pure $ randomRM (0, noise) g
  cluster <- randomM g
  let !r = case cluster of
        Cluster0 -> 1.0
        Cluster1 -> -1.0
      !mag = (1.0 - coreR) * theta / maxTheta + coreR
      coord = P $ r * mag *^ V2 (-cos theta) (sin theta) ^+^ dev
  pure ClusteredPoint {..}
  where
    !coreR = 1.5 * noise
    !maxTheta = 780 * 2 * pi / 360

clusteredPointD ::
  ( Dia.V b ~ V2
  , Dia.N b ~ Double
  , Dia.Renderable (Dia.Path V2 Double) b
  ) =>
  ClusteredPoint ->
  Dia.Diagram b
clusteredPointD ClusteredPoint {..} =
  let color = case cluster of
        Cluster0 -> Dia.red
        Cluster1 -> Dia.blue
   in Dia.circle 0.025 & Dia.lcA Dia.transparent & Dia.fc color & Dia.moveTo coord

drawClusteredPoints ::
  ( Metric (Dia.V b)
  , Floating (Dia.N b)
  , Ord (Dia.N b)
  , Dia.Renderable (Dia.Path V2 Double) b
  , Dia.N b ~ Double
  , Dia.V b ~ V2
  ) =>
  U.Vector ClusteredPoint ->
  Dia.QDiagram b (Dia.V b) (Dia.N b) Dia.Any
drawClusteredPoints =
  mconcat
    [ foldMapOf vectorTraverse clusteredPointD
    , const $ Dia.square 2 & Dia.opacity 0.0
    ]
    >>> Dia.centerXY
    >>> Dia.pad 1.25
