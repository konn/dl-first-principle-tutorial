{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RankNTypes #-}

module Numeric.Optimisation (gradDescent) where

import Data.List (iterate')
import Data.Reflection (Reifies)
import Linear (Additive ((^-^)), (*^))
import Numeric.AD (grad)
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Type.Reflection (Typeable)

-- | A simple gradient desecent with a static learning rate
gradDescent ::
  (Additive f, Traversable f, Num a) =>
  -- | Learning rate
  a ->
  (forall s. (Reifies s Tape, Typeable s) => f (Reverse s a) -> Reverse s a) ->
  f a ->
  [f a]
{-# INLINE gradDescent #-}
gradDescent gamma f = iterate' $ \x -> x ^-^ gamma *^ grad f x
