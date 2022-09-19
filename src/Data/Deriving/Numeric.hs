{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveTraversable #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Data.Deriving.Numeric (GenericNum (..)) where

import Data.Coerce (coerce)
import Data.Function
import GHC.Generics

newtype GenericNum a = GenericNum {unGenericNum :: a}
  deriving (Show, Eq, Ord, Generic, Generic1, Functor, Foldable, Traversable)

instance (Generic a, GNum (Rep a)) => Num (GenericNum a) where
  (+) = fmap (GenericNum . to) . (gadd `on` (from . unGenericNum))
  {-# INLINE (+) #-}
  (-) = fmap (GenericNum . to) . (gsub `on` (from . unGenericNum))
  {-# INLINE (-) #-}
  (*) = fmap (GenericNum . to) . (gmul `on` (from . unGenericNum))
  {-# INLINE (*) #-}
  abs = GenericNum . to . gabs . from . unGenericNum
  {-# INLINE abs #-}
  signum = GenericNum . to . gsignum . from . unGenericNum
  {-# INLINE signum #-}
  fromInteger = GenericNum . to . gfromInteger
  {-# INLINE fromInteger #-}
  negate = GenericNum . to . gneg . from . unGenericNum
  {-# INLINE negate #-}

instance (Generic a, GFractional (Rep a)) => Fractional (GenericNum a) where
  fromRational = GenericNum . to . gfromRational
  {-# INLINE fromRational #-}
  recip = GenericNum . to . grecip . from . unGenericNum
  {-# INLINE recip #-}
  (/) = fmap (GenericNum . to) . (gdiv `on` (from . unGenericNum))
  {-# INLINE (/) #-}

instance (Generic a, GFloating (Rep a)) => Floating (GenericNum a) where
  pi = GenericNum $ to gpi
  {-# INLINE pi #-}
  exp = GenericNum . to . gexp . from . unGenericNum
  {-# INLINE exp #-}
  log = GenericNum . to . glog . from . unGenericNum
  {-# INLINE log #-}
  sin = GenericNum . to . gsin . from . unGenericNum
  {-# INLINE sin #-}
  cos = GenericNum . to . gcos . from . unGenericNum
  {-# INLINE cos #-}
  tan = GenericNum . to . gtan . from . unGenericNum
  {-# INLINE tan #-}
  asin = GenericNum . to . gasin . from . unGenericNum
  {-# INLINE asin #-}
  acos = GenericNum . to . gacos . from . unGenericNum
  {-# INLINE acos #-}
  atan = GenericNum . to . gatan . from . unGenericNum
  {-# INLINE atan #-}
  sinh = GenericNum . to . gsinh . from . unGenericNum
  {-# INLINE sinh #-}
  cosh = GenericNum . to . gcosh . from . unGenericNum
  {-# INLINE cosh #-}
  tanh = GenericNum . to . gtanh . from . unGenericNum
  {-# INLINE tanh #-}
  asinh = GenericNum . to . gasinh . from . unGenericNum
  {-# INLINE asinh #-}
  acosh = GenericNum . to . gacosh . from . unGenericNum
  {-# INLINE acosh #-}
  atanh = GenericNum . to . gatanh . from . unGenericNum
  {-# INLINE atanh #-}

class GNum f where
  gfromInteger :: Integer -> f ()
  gadd :: f () -> f () -> f ()
  gsub :: f () -> f () -> f ()
  gmul :: f () -> f () -> f ()
  gneg :: f () -> f ()
  gsignum :: f () -> f ()
  gabs :: f () -> f ()

class GNum f => GFractional f where
  gfromRational :: Rational -> f ()
  gdiv :: f () -> f () -> f ()
  grecip :: f () -> f ()

class (GFractional f) => GFloating f where
  gpi :: f ()
  gexp :: f () -> f ()
  glog :: f () -> f ()
  gsin :: f () -> f ()
  gcos :: f () -> f ()
  gtan :: f () -> f ()
  gasin :: f () -> f ()
  gacos :: f () -> f ()
  gatan :: f () -> f ()
  gsinh :: f () -> f ()
  gcosh :: f () -> f ()
  gtanh :: f () -> f ()
  gasinh :: f () -> f ()
  gacosh :: f () -> f ()
  gatanh :: f () -> f ()

instance GNum f => GNum (M1 i c f) where
  gfromInteger = coerce $ gfromInteger @f
  gadd = coerce $ gadd @f
  {-# INLINE gadd #-}
  gsub = coerce $ gsub @f
  {-# INLINE gsub #-}
  gmul = coerce $ gmul @f
  {-# INLINE gmul #-}
  gneg = coerce $ gneg @f
  {-# INLINE gneg #-}
  gsignum = coerce $ gsignum @f
  {-# INLINE gsignum #-}
  gabs = coerce $ gabs @f
  {-# INLINE gabs #-}

instance GFractional f => GFractional (M1 i c f) where
  gfromRational = coerce $ gfromRational @f
  {-# INLINE gfromRational #-}
  gdiv = coerce $ gdiv @f
  {-# INLINE gdiv #-}
  grecip = coerce $ grecip @f
  {-# INLINE grecip #-}

instance GFloating f => GFloating (M1 i c f) where
  gpi = coerce $ gpi @f
  {-# INLINE gpi #-}
  gexp = coerce $ gexp @f
  {-# INLINE gexp #-}
  glog = coerce $ glog @f
  {-# INLINE glog #-}
  gsin = coerce $ gsin @f
  {-# INLINE gsin #-}
  gcos = coerce $ gcos @f
  {-# INLINE gcos #-}
  gtan = coerce $ gtan @f
  {-# INLINE gtan #-}
  gasin = coerce $ gasin @f
  {-# INLINE gasin #-}
  gacos = coerce $ gacos @f
  {-# INLINE gacos #-}
  gatan = coerce $ gatan @f
  {-# INLINE gatan #-}
  gsinh = coerce $ gsinh @f
  {-# INLINE gsinh #-}
  gcosh = coerce $ gcosh @f
  {-# INLINE gcosh #-}
  gtanh = coerce $ gtanh @f
  {-# INLINE gtanh #-}
  gasinh = coerce $ gasinh @f
  {-# INLINE gasinh #-}
  gacosh = coerce $ gacosh @f
  {-# INLINE gacosh #-}
  gatanh = coerce $ gatanh @f
  {-# INLINE gatanh #-}

instance Num c => GNum (K1 i c) where
  gfromInteger = coerce $ fromInteger @c
  {-# INLINE gfromInteger #-}
  gadd = coerce $ (+) @c
  {-# INLINE gadd #-}
  gsub = coerce $ (-) @c
  {-# INLINE gsub #-}
  gmul = coerce $ (*) @c
  {-# INLINE gmul #-}
  gneg = coerce $ negate @c
  {-# INLINE gneg #-}
  gsignum = coerce $ signum @c
  {-# INLINE gsignum #-}
  gabs = coerce $ abs @c
  {-# INLINE gabs #-}

instance Fractional c => GFractional (K1 i c) where
  gfromRational = coerce $ fromRational @c
  {-# INLINE gfromRational #-}
  gdiv = coerce $ (/) @c
  {-# INLINE gdiv #-}
  grecip = coerce $ recip @c
  {-# INLINE grecip #-}

instance Floating c => GFloating (K1 i c) where
  gpi = coerce $ pi @c
  {-# INLINE gpi #-}
  gexp = coerce $ exp @c
  {-# INLINE gexp #-}
  glog = coerce $ log @c
  {-# INLINE glog #-}
  gsin = coerce $ sin @c
  {-# INLINE gsin #-}
  gcos = coerce $ cos @c
  {-# INLINE gcos #-}
  gtan = coerce $ tan @c
  {-# INLINE gtan #-}
  gasin = coerce $ asin @c
  {-# INLINE gasin #-}
  gacos = coerce $ acos @c
  {-# INLINE gacos #-}
  gatan = coerce $ atan @c
  {-# INLINE gatan #-}
  gsinh = coerce $ sinh @c
  {-# INLINE gsinh #-}
  gcosh = coerce $ cosh @c
  {-# INLINE gcosh #-}
  gtanh = coerce $ tanh @c
  {-# INLINE gtanh #-}
  gasinh = coerce $ asinh @c
  {-# INLINE gasinh #-}
  gacosh = coerce $ acosh @c
  {-# INLINE gacosh #-}
  gatanh = coerce $ atanh @c
  {-# INLINE gatanh #-}

instance GNum U1 where
  gfromInteger = mempty
  {-# INLINE gfromInteger #-}
  gadd = mempty
  {-# INLINE gadd #-}
  gsub = mempty
  {-# INLINE gsub #-}
  gmul = mempty
  {-# INLINE gmul #-}
  gneg = mempty
  {-# INLINE gneg #-}
  gsignum = mempty
  {-# INLINE gsignum #-}
  gabs = mempty
  {-# INLINE gabs #-}

instance GFractional U1 where
  gfromRational = mempty
  {-# INLINE gfromRational #-}
  gdiv = mempty
  {-# INLINE gdiv #-}
  grecip = mempty
  {-# INLINE grecip #-}

instance GFloating U1 where
  gpi = mempty
  {-# INLINE gpi #-}
  gexp = mempty
  {-# INLINE gexp #-}
  glog = mempty
  {-# INLINE glog #-}
  gsin = mempty
  {-# INLINE gsin #-}
  gcos = mempty
  {-# INLINE gcos #-}
  gtan = mempty
  {-# INLINE gtan #-}
  gasin = mempty
  {-# INLINE gasin #-}
  gacos = mempty
  {-# INLINE gacos #-}
  gatan = mempty
  {-# INLINE gatan #-}
  gsinh = mempty
  {-# INLINE gsinh #-}
  gcosh = mempty
  {-# INLINE gcosh #-}
  gtanh = mempty
  {-# INLINE gtanh #-}
  gasinh = mempty
  {-# INLINE gasinh #-}
  gacosh = mempty
  {-# INLINE gacosh #-}
  gatanh = mempty
  {-# INLINE gatanh #-}

instance (GNum f, GNum g) => GNum (f :*: g) where
  gfromInteger = (:*:) <$> gfromInteger <*> gfromInteger
  gadd = liftP2 gadd gadd
  {-# INLINE gadd #-}
  gsub = liftP2 gsub gsub
  {-# INLINE gsub #-}
  gmul = liftP2 gmul gmul
  {-# INLINE gmul #-}
  gneg = liftP1 gneg gneg
  {-# INLINE gneg #-}
  gsignum = liftP1 gsignum gsignum
  {-# INLINE gsignum #-}
  gabs = liftP1 gabs gabs
  {-# INLINE gabs #-}

instance (GFractional f, GFractional g) => GFractional (f :*: g) where
  gfromRational = (:*:) <$> gfromRational <*> gfromRational
  {-# INLINE gfromRational #-}
  gdiv = liftP2 gdiv gdiv
  {-# INLINE gdiv #-}
  grecip = liftP1 grecip grecip
  {-# INLINE grecip #-}

instance (GFloating f, GFloating g) => GFloating (f :*: g) where
  gpi = gpi :*: gpi
  {-# INLINE gpi #-}
  gexp = liftP1 gexp gexp
  {-# INLINE gexp #-}
  glog = liftP1 glog glog
  {-# INLINE glog #-}
  gsin = liftP1 gsin gsin
  {-# INLINE gsin #-}
  gcos = liftP1 gcos gcos
  {-# INLINE gcos #-}
  gtan = liftP1 gtan gtan
  {-# INLINE gtan #-}
  gasin = liftP1 gasin gasin
  {-# INLINE gasin #-}
  gacos = liftP1 gacos gacos
  {-# INLINE gacos #-}
  gatan = liftP1 gatan gatan
  {-# INLINE gatan #-}
  gsinh = liftP1 gsinh gsinh
  {-# INLINE gsinh #-}
  gcosh = liftP1 gcosh gcosh
  {-# INLINE gcosh #-}
  gtanh = liftP1 gtanh gtanh
  {-# INLINE gtanh #-}
  gasinh = liftP1 gasinh gasinh
  {-# INLINE gasinh #-}
  gacosh = liftP1 gacosh gacosh
  {-# INLINE gacosh #-}
  gatanh = liftP1 gatanh gatanh
  {-# INLINE gatanh #-}

liftP1 :: (f a -> f b) -> (g a -> g b) -> (f :*: g) a -> (f :*: g) b
{-# INLINE liftP1 #-}
liftP1 f g = \case (l :*: r) -> (f l :*: g r)

liftP2 :: (f a -> f b -> f c) -> (g a -> g b -> g c) -> (f :*: g) a -> (f :*: g) b -> (f :*: g) c
{-# INLINE liftP2 #-}
liftP2 f g = \case (l :*: r) -> \case (l' :*: r') -> (f l l' :*: g r r')
