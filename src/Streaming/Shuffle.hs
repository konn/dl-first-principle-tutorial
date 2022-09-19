{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Streaming.Shuffle (shuffleBuffered, shuffleBuffered', shuffleStreamL) where

import Control.Arrow
import qualified Control.Foldl as L
import Control.Monad ((>=>))
import Control.Monad.Trans.Class (lift)
import Data.Functor.Of (Of (..))
import qualified Data.Heap as H
import qualified Data.Vector.Generic as G
import qualified Data.Vector.Generic.Mutable as MG
import qualified Data.Vector.Unboxed as U
import qualified Streaming as SS
import Streaming.Prelude (Stream)
import qualified Streaming.Prelude as S
import System.Random.Stateful (RandomGenM, randomRM, uniformRM)

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

-- | A variant using priority queue
shuffleBuffered' ::
  (RandomGenM g r m) =>
  g ->
  Int ->
  Stream (Of a) m x ->
  Stream (Of a) m ()
shuffleBuffered' g n =
  SS.hoist lift
    >>> L.impurely S.foldM_ (shuffleStreamL g n)

shuffleStreamL ::
  (RandomGenM g r m) =>
  g ->
  Int ->
  L.FoldM (Stream (Of a) m) a ()
shuffleStreamL g n = L.FoldM step (pure H.empty) (S.map H.payload . S.each)
  where
    step !h !a = do
      w <- lift $ uniformRM (0.0 :: Double, 1.0) g
      let h' = H.insert H.Entry {H.priority = w, H.payload = a} h
          (h'', mout)
            | H.size h' <= n = (h', Nothing)
            | Just (over, rest) <- H.uncons h' = (rest, Just $ H.payload over)
            | otherwise = (h', Nothing)
      S.each mout
      pure h''

shuffleV :: (G.Vector v a, RandomGenM g r m) => g -> v a -> m (v a)
shuffleV g v = do
  let n = G.length v
  js <- U.generateM n $ \i -> randomRM (0, i) g
  pure $ G.modify (U.iforM_ js . MG.swap) v
