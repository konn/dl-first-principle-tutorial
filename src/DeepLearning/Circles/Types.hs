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
    ClusteredPoint (..),
    Cluster (..),
    clusteredPointD,
    drawClusteredPoints,
  )
where

import Control.Arrow ((>>>))
import Control.Lens (foldMapOf, (^.))
import Control.Monad (guard)
import Control.Monad.Fix (fix)
import Data.Bit.ThreadSafe (Bit (..))
import Data.Csv (ToRecord (..))
import qualified Data.Csv as Csv
import Data.Function ((&))
import qualified Data.Vector as V
import Data.Vector.Generic.Lens
import qualified Data.Vector.Unboxed as U
import Data.Vector.Unboxed.Deriving (derivingUnbox)
import Diagrams.Prelude (Diagram, bg, blue, centerXY, fc, lcA, moveTo, opacity, pad, red, square, transparent, white)
import qualified Diagrams.Prelude as Dia
import GHC.Generics (Generic)
import Linear
import Linear.Affine
import System.Random.Stateful

data Cluster = Cluster0 | Cluster1
  deriving (Show, Eq, Ord, Generic, Enum, Bounded)

derivingUnbox
  "Cluster"
  [t|Cluster -> Bit|]
  [|\case Cluster0 -> Bit False; Cluster1 -> Bit True|]
  [|\(Bit p) -> if p then Cluster1 else Cluster0|]

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
dualCircles :: RandomGenM g r m => g -> Int -> m (U.Vector ClusteredPoint)
dualCircles g n = do
  r1 <- randomRM (0.05, 1.0) g
  r2 <- fix $ \retry -> do
    r2 <- randomRM (0.05, 1.0) g
    if abs (r1 - r2) > 0.1
      then pure r2
      else retry
  let dr =
        minimum
          [ abs (r1 - r2) / 10
          , 0.05
          , r1 / 2
          , r2 / 2
          ]
  U.replicateM n $ do
    p <- randomM g
    dev <- randomRM (-dr, dr) g
    let (r, cluster)
          | p = (r1, Cluster0)
          | otherwise = (r2, Cluster1)
    theta <- randomRM (-pi, pi) g
    let coord = P $ (r + dev) *^ V2 (cos theta) (sin theta)
    pure ClusteredPoint {..}

clusteredPointD ::
  ( Dia.V b ~ V2
  , Dia.N b ~ Double
  , Dia.Renderable (Dia.Path V2 Double) b
  ) =>
  ClusteredPoint ->
  Diagrams.Prelude.Diagram b
clusteredPointD ClusteredPoint {..} =
  let color = case cluster of
        Cluster0 -> Diagrams.Prelude.red
        Cluster1 -> Diagrams.Prelude.blue
   in Dia.circle 0.025 & Diagrams.Prelude.lcA Diagrams.Prelude.transparent & Diagrams.Prelude.fc color & Diagrams.Prelude.moveTo coord

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
    , const $ Diagrams.Prelude.square 2 & Diagrams.Prelude.opacity 0.0
    ]
    >>> Diagrams.Prelude.centerXY
    >>> Diagrams.Prelude.pad 1.25
    >>> Diagrams.Prelude.bg Diagrams.Prelude.white
