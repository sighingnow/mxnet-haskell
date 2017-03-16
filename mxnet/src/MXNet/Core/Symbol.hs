-----------------------------------------------------------
-- |
-- module:                      MXNet.Core.Symbol
-- copyright:                   (c) 2016 Tao He
-- license:                     MIT
-- maintainer:                  sighingnow@gmail.com
--
-- Symbol module.
--
{-# OPTIONS_GHC -Wno-name-shadowing #-}
{-# OPTIONS_GHC -Wno-redundant-constraints #-}
{-# LANGUAGE OverloadedLists #-}
{-# LANGUAGE RecordWildCards #-}

module MXNet.Core.Symbol where

import           Control.Exception
import           Control.Monad
import           Data.HashMap.Strict (HashMap)
import qualified Data.HashMap.Strict as HM
import           Data.Monoid
import           Foreign.Ptr (nullPtr)
import           System.IO.Unsafe
import           Unsafe.Coerce (unsafeCoerce)

import           MXNet.Core.Base
import           MXNet.Core.Executor hiding (getHandle)
import           MXNet.Core.NDArray hiding (getHandle)
import qualified MXNet.Core.NDArray as NDArray (getHandle)
import           MXNet.Core.NNVM.Base

-- | Type alias for variable.
newtype Symbol a = Symbol { getHandle :: SymbolHandle }

instance DType a => Show (Symbol a) where
    show sym = unsafePerformIO $ do
        (_, str) <- mxSymbolPrint (getHandle sym)
        return str

-- | Make a new symbolic variable with given name.
makeVar :: DType a
        => String           -- ^ Name.
        -> IO (Symbol a)    -- ^ Result variable.
makeVar = mxSymbolCreateVariable >=> return . Symbol . snd

-- | Get the name of a given variable.
name :: DType a => Symbol a -> IO String
name = mxSymbolGetName . getHandle >=> \(_, nm, _) -> return nm

-- | Get some attribute of symbol.
attr :: DType a => Symbol a -> String -> IO (Maybe String)
attr sym key = do
    (_, s, success) <- mxSymbolGetAttr (getHandle sym) key
    return $ if success == 0    -- 0 when success, -1 when failure happens
                then Just s
                else Nothing

-- | Infer the type of the given symbol, return arg_types, out_types and aux_types.
inferType :: DType a => Symbol a -> [String] -> IO ([Int], [Int], [Int])
inferType sym args = do
    (_, arg, out, aux) <- mxSymbolInferType (getHandle sym) args
    return (arg, out, aux)

-- | Infer the shape of the given symbol, return the in, out and auxiliary shape size.
inferShape :: DType a => Symbol a -> [String] -> IO ([[Int]], [[Int]], [[Int]])
inferShape sym args = do
    (_, arg, out, aux) <- mxSymbolInferShape (getHandle sym) args [0] []
    return (arg, out, aux)

-- | List all input arguments.
getInputs :: DType a => Symbol a -> IO [String]
getInputs sym = snd <$> mxSymbolListArguments (getHandle sym)

-- | List all output results.
getOutputs :: DType a => Symbol a -> IO [String]
getOutputs sym = snd <$> mxSymbolListOutputs (getHandle sym)

-- | List all auxiliary states.
getAuxiliary :: DType a => Symbol a -> IO [String]
getAuxiliary sym = snd <$> mxSymbolListAuxiliaryStates (getHandle sym)

-- | Get the autodiff of current symbol.
-- This function can only be used if current symbol is a loss function.
grad :: DType a => Symbol a -> [String] -> IO (Symbol a)
grad sym args = do
    let nargs = fromIntegral (length args)
    (_, handle) <- mxSymbolGrad (getHandle sym) nargs args
    return $ Symbol handle

-- | Bind with explicit argument mapping (name -- value mapping).
bind :: DType a
     => Symbol a
     -> Context
     -> HashMap String (NDArray a)
     -> IO (Executor a)
bind sym Context{..} args = do
    inputs <- genNDArrayMapping args <$> getInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    (_, exec) <- mxExecutorBind (getHandle sym)
                                deviceType
                                deviceId
                                (fromIntegral (HM.size inputs))     -- length of input arguments.
                                (NDArray.getHandle <$> HM.elems inputs)
                                (replicate (HM.size inputs) (unsafeCoerce nullPtr))
                                req_types
                                0                                   -- length of auxiliary states.
                                []                                  -- no auxiliary states.
    return $ Executor exec
  where
    -- | Get ndarray lists handles from input arguments.
    genNDArrayMapping args arg_names = HM.fromList (genfn <$> arg_names)
      where
        genfn nm = case HM.lookup nm args of
                        Just v -> (nm, v)
                        Nothing -> throw . userError $ "getNDArrayInputs: no argument " <> nm

-- | Bind without explicit argument mapping (name -- value mapping).
bind' :: DType a
      => Symbol a
      -> Context
      -> [NDArray a]
      -> IO (Executor a)
bind' sym Context{..} args = do
    inputs <- genNDArrayMapping args <$> getInputs sym
    -- req_map = {'null': 0, 'write': 1, 'add': 3}
    let req_types = replicate (HM.size inputs) 1        -- use default value.
    (_, exec) <- mxExecutorBind (getHandle sym)
                                deviceType
                                deviceId
                                (fromIntegral (HM.size inputs))     -- length of input arguments.
                                (NDArray.getHandle <$> HM.elems inputs)
                                (replicate (HM.size inputs) (unsafeCoerce nullPtr))
                                req_types
                                0                                   -- length of auxiliary states.
                                []                                  -- no auxiliary states.
    return $ Executor exec
  where
    -- | Get ndarray lists handles from input arguments without explicit argument names.
    genNDArrayMapping args names =
        assert (length args == length names) $
            HM.fromList (zip names args)
