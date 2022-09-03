module Data.Format.SpaceSeparated (decodeSSV) where

import qualified Data.ByteString.Lazy as LBS
import Data.Csv
import Data.Generics.Labels ()
import qualified Data.Vector as V

decodeSSV :: FromRecord a => LBS.ByteString -> Either String (V.Vector a)
decodeSSV =
  decodeWith
    defaultDecodeOptions {decDelimiter = 32} -- half-width space
    NoHeader
