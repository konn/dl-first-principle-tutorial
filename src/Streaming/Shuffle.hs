{-# LANGUAGE TypeApplications #-}

module Streaming.Shuffle (shuffleBuffered) where

import Control.Arrow
import qualified Control.Foldl as L
import Control.Monad ((>=>))
import Data.Functor.Of (Of (..))
import qualified Data.Vector.Generic as G
import qualified Data.Vector.Generic.Mutable as MG
import qualified Data.Vector.Unboxed as U
import qualified Streaming as SS
import Streaming.Prelude (Stream)
import qualified Streaming.Prelude as S
import System.Random.Stateful (RandomGenM, randomRM)

-- | Chunking the input stream into the same size, and then applies shuffle to each chunks.
shuffleBuffered ::
  (Monad m, U.Unbox a, RandomGenM g r m) =>
  g ->
  Int ->
  Stream (Of a) m x ->
  Stream (Of a) m x
shuffleBuffered gen n =
  SS.chunksOf n
    >>> S.mapped
      ( L.impurely S.foldM (L.generalize $ L.vector @U.Vector)
          >=> \(vec :> x) -> do
            vec' <- shuffleV gen vec
            pure $ x <$ U.mapM_ S.yield vec'
      )
    >>> SS.concats

shuffleV :: (G.Vector v a, RandomGenM g r m) => g -> v a -> m (v a)
shuffleV g v = do
  let n = G.length v
  js <- U.generateM n $ \i -> randomRM (0, i) g
  pure $ G.modify (U.iforM_ js . MG.swap) v
