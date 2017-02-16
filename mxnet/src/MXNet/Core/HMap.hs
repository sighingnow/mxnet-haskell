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
-- > let a = add @"a" (1 :: Int) empty
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
    ( HMap
    , empty
    , add
    , (.+.)
    , get
    , (.->.)
    , update
    , set
    , dump
    ) where

import           Data.Proxy (Proxy (..))
import           GHC.TypeLits
import           Data.List (intercalate)
import           Data.Monoid ((<>))

data KV v = Symbol := v

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

-- | If a KVList is a sublist of another KVList.
data IfSubType = SubType | NotSubType

-- | Make a IfSubType from IfHasKey.
type family MkIfSubType (r :: IfHasKey) :: IfSubType where
    MkIfSubType ('Yes _) = 'SubType
    MkIfSubType 'No = 'NotSubType

-- | Combine two IfSubType type.
type family AllSubType (t1:: IfSubType) (t2 :: IfSubType) :: IfSubType where
    AllSubType 'SubType 'SubType = 'SubType
    AllSubType _ _ = 'NotSubType

-- | If the first KVList is part of the second KVList.
type family MatchKVList (kvs1 :: [KV *]) (kvs2 :: [KV *]) :: IfSubType where
    MatchKVList '[] _ = 'SubType
    MatchKVList (k1 ':= v1 ': kvs') kvs2 = AllSubType (MkIfSubType (FindKV k1 v1 kvs2)) (MatchKVList kvs' kvs2)

-- | Constraint ensure 'HMap' must contain k-v pair.
--
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
empty :: HMap '[]
empty = HMap Nil

{-# INLINE empty #-}

-- | Add a key-value pair into the HMap (via TypeApplications).
add :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
add v (HMap kvs) = HMap (Cons v kvs)

{-# INLINE add #-}

-- | Infix version of @add@.
(.+.) :: forall k v kvs. 'No ~ FindKV k v kvs => v -> HMap kvs -> HMap (k ':= v ': kvs)
(.+.) = add

{-# INLINE (.+.) #-}

-- | Get the value of an existing key.
get :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
get = get' @k

{-# INLINE get #-}

-- | Infix version of @get@.
(.->.) :: forall (k :: Symbol) v kvs. InDict k v kvs => HMap kvs -> v
(.->.) = get @k

{-# INLINE (.->.) #-}

-- | Update the value of an existing key.
update :: forall (k :: Symbol) v kvs. InDict k v kvs => (v -> v) -> HMap kvs -> HMap kvs
update = update' @k

{-# INLINE update #-}

-- | Set the value of an existing key.
set :: forall k v kvs. InDict k v kvs => v -> HMap kvs -> HMap kvs
set v = update' @k (const v)

{-# INLINE set #-}

class ShowKV (kvs :: [KV *]) where
    show' :: forall k v. KVList kvs -> [(String, String)]

instance ShowKV '[] where
    show' _ = []
    {-# INLINE show' #-}

instance (KnownSymbol k, Show v, ShowKV kvs) => ShowKV (k ':= v ': kvs) where
    show' (Cons v kvs') = showImpl v : show' kvs'
        where showImpl value = (symbolVal (Proxy :: Proxy k), show value)
    {-# INLINE show' #-}

instance ShowKV kvs => Show (HMap kvs) where
    show m = "[" <> (intercalate ", " . map (\(k, v) -> k <> " = " <> v) . show' . getKVList $ m) <> "]"

-- | Dump key-value pair in HMap as [(k, v)].
dump :: forall kvs. ShowKV kvs => HMap kvs -> [(String, String)]
dump = show' . getKVList

{-# INLINE dump #-}
