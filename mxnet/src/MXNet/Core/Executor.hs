-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Symbol
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# OPTIONS_GHC -Wno-redundant-constraints #-}

module MXNet.Core.Executor where

import           Control.Monad
import           MXNet.Core.Base
import           MXNet.Core.NDArray (NDArray(NDArray), DType)

-- | Type alias for variable.
newtype Executor a = Executor { getHandle :: ExecutorHandle }

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

getOutput :: DType a
          => Executor a
          -> IO [NDArray a]
getOutput exec = do
    (_, outs) <- mxExecutorOutputs (getHandle exec)
    return $ NDArray <$> outs
