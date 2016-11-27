-----------------------------------------------------------
-- |
-- module:                      C2HS.C.Extra
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Convenient marshallers for complicate C types.
--
module C2HS.C.Extra
    ( peekIntegral
    , peekString
    ) where

import Foreign.C.String ( peekCString )
import Foreign.C.Types ( CChar, CInt )
import Foreign.Ptr ( Ptr )
import Foreign.Storable ( Storable, peek )

-- | Peek from pointer then cast to another integral type.
peekIntegral :: (Integral a, Storable a, Integral b) => Ptr a -> IO b
peekIntegral = (fromIntegral <$>) . peek

{-# INLINE peekIntegral #-}
{-# SPECIALIZE peekIntegral :: Ptr CInt -> IO Int #-}

-- | Peek string from a two-dimension pointer of CChar.
peekString :: Ptr (Ptr CChar) -> IO String
peekString p = peek p >>= peekCString

{-# INLINE peekString #-}
