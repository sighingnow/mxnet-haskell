-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.HMap
-- copyright:                   (c) 2016-2017 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Updatable heterogeneous map.
--
-- @
-- > let a = add @"a" (1 :: Int) nil
-- > a
-- [a = 1]
-- > let b = update @"a" (+1) a
-- > b
-- [a = 2]
-- > let c = add @"b" (Nothing :: Maybe Float) b
-- > c
-- [b = Nothing, a = 2]
-- > set @"b" (Just 128) c
-- [b = Just 128.0, a = 2]
-- @
--
{-# OPTIONS_GHC -Wno-unused-foralls #-}
{-# OPTIONS_GHC -Wno-unused-type-patterns #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}

{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module MXNet.Core.HMap
    ( -- * HMap type definition
      HMap
      -- * Type level constraints and operators
    , KV (..)
    , ShowKV (..)
    , MatchKVList (..)
      -- * Operations on HMap.
    , nil
    , add
    , add'
    , (.+.)
    , get
    , (.->.)
    , update
    , set
    , mergeTo
    , dump
    ) where

import           GHC.TypeLits
import           Data.List (intercalate)
import           Data.Monoid ((<>))
import           Data.Proxy (Proxy (..))
import           Data.Typeable (Typeable, typeOf)

data KV v = Symbol := v

infixr 6 :=

data KVList (kvs :: [KV *]) where
    Nil :: KVList '[]
    Cons :: v -> KVList kvs -> KVList (k ':= v ': kvs)

-- | If a KVList has a specified type of KV pair.
data IfHasKey = Yes Symbol | No

-- | Find specified key-value type pair in KVList.
type family FindKV (k :: Symbol) v (kvs :: [KV *]) :: IfHasKey where
    FindKV k _ '[] = 'No
    FindKV k v (k ':= v ': kvs) = 'Yes k
    FindKV k1 v1 (k2 ':= v2 ': kvs) = FindKV k1 v1 kvs

-- | HMap definition.
newtype HMap (kvs :: [KV *]) = HMap { getKVList :: KVList kvs }

-- | Constraint ensure 'HMap' must contain k-v pair.
class InDict (k :: Symbol) (v :: *) (kvs :: [KV *]) | k kvs -> v where
    get' :: HMap kvs -> v
    update' :: (v -> v) -> HMap kvs -> HMap kvs

instance {-# OVERLAPPING #-} InDict k v (k ':= v ': kvs) where
    get' (HMap (Cons v _)) = v
    {-# INLINE get' #-}
    update' f (HMap (Cons v kvs)) = HMap $ Cons (f v) kvs
    {-# INLINE update' #-}

instance (InDict k v kvs, 'Yes k ~ FindKV k v (k' ':= v' ': kvs)) => InDict k v (k' ':= v' ': kvs) where
    get' (HMap (Cons _ kvs)) =  get' @k (HMap kvs)
    {-# INLINE get' #-}
    update' f (HMap (Cons v kvs)) = HMap $ Cons v (getKVList $ update' @k f (HMap kvs))
    {-# INLINE update' #-}

-- | Create an empty HMap.
nil :: HMap '[]
nil = HMap Nil

{-# INLINE nil #-}

-- | Add a key-value pair into the HMap (via TypeApplications).
add :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
add v (HMap kvs) = HMap (Cons v kvs)

{-# INLINE add #-}

-- | Add a key-value pair into the HMap (via TypeApplications).
--
-- FIXME should have a @'No ~ FindKV k v kvs@ constraint here.
add' :: forall k v kvs. Proxy k -> v -> HMap kvs -> HMap (k ':= v ': kvs)
add' _ v (HMap kvs) = HMap (Cons v kvs)

{-# INLINE add' #-}


-- | Infix version of @add@.
(.+.) :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
(.+.) = add

infix 8 .+.

{-# INLINE (.+.) #-}

-- | Get the value of an existing key.
get :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
get = get' @k

{-# INLINE get #-}

-- | Infix version of @get@.
(.->.) :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
(.->.) = get @k

infix 7 .->.

{-# INLINE (.->.) #-}

-- | Update the value of an existing key.
update :: forall (k :: Symbol) v kvs. InDict k v kvs => (v -> v) -> HMap kvs -> HMap kvs
update = update' @k

{-# INLINE update #-}

-- | Set the value of an existing key.
set :: forall k v kvs. InDict k v kvs => v -> HMap kvs -> HMap kvs
set v = update' @k (const v)

{-# INLINE set #-}

-- | Merge the first KVList into the second one.
class MatchKVList (kvs1 :: [KV *]) (kvs2 :: [KV *]) where
    -- | Update all values in the first HMap into the second KVList.
    mergeTo' :: HMap kvs1 -> HMap kvs2 -> HMap kvs2

instance MatchKVList ('[]) (kvs2) where
    mergeTo' _ m2 = m2

instance (MatchKVList kvs1 kvs2, InDict k v kvs2) => MatchKVList (k ':= v ': kvs1) kvs2 where
    mergeTo' (HMap (Cons v kvs)) m2 = mergeTo' (HMap kvs) (set @k v m2)

-- | Update all values in the first HMap into the second KVList.
mergeTo :: forall (kvs1 :: [KV *]) (kvs2 :: [KV *]). MatchKVList kvs1 kvs2 => HMap kvs1 -> HMap kvs2 -> HMap kvs2
mergeTo = mergeTo'

class ShowKV (kvs :: [KV *]) where
    show' :: forall k v. KVList kvs -> [(String, String)]

instance ShowKV '[] where
    show' _ = []
    {-# INLINE show' #-}

instance (KnownSymbol k, Typeable v, Show v, ShowKV kvs) => ShowKV (k ':= v ': kvs) where
    show' (Cons v kvs') = showImpl v : show' kvs'
        where showImpl value = (symbolVal (Proxy :: Proxy k), if typeOf value == typeOf "" then (init . tail . show)  value else show value) -- special rule for string value.
    {-# INLINE show' #-}

instance ShowKV kvs => Show (HMap kvs) where
    show m = "[" <> (intercalate ", " . map (\(k, v) -> k <> " = " <> v) . show' . getKVList $ m) <> "]"
    {-# INLINE show #-}

-- | Dump key-value pair in HMap as [(k, v)].
dump :: forall kvs. ShowKV kvs => HMap kvs -> [(String, String)]
dump = show' . getKVList

{-# INLINE dump #-}
