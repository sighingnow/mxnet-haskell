-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Base.Executor
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# LANGUAGE DeriveGeneric #-}

module MXNet.Core.Base.Executor where

import           Control.Monad
import           GHC.Generics
import           MXNet.Core.Base.Internal
import           MXNet.Core.Base.DType
import           MXNet.Core.Base.NDArray (NDArray(NDArray))

-- | Type alias for variable.
newtype Executor a = Executor { getHandle :: ExecutorHandle }
    deriving Generic

-- | Make an executor using the given handler.
makeExecutor :: DType a
             => ExecutorHandle
             -> IO (Executor a)
makeExecutor = return . Executor

-- | Executor forward method.
forward :: DType a
        => Executor a   -- ^ The executor handle.
        -> Bool         -- ^ Whether this forward is for evaluation purpose.
        -> IO ()
forward exec train = void $ mxExecutorForward (getHandle exec) (if train then 1 else 0)

-- | Executor backward method.
backward :: DType a
         => Executor a  -- ^ The executor handle.
         -> IO ()
backward exec = void $ mxExecutorBackward (getHandle exec) 0 []

getOutputs :: DType a
          => Executor a
          -> IO [NDArray a]
getOutputs exec = do
    outs <- mxExecutorOutputs (getHandle exec)
    return $ NDArray <$> outs
