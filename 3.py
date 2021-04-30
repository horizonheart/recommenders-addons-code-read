from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import itertools
import math
import numpy as np
import os

from tensorflow_recommenders_addons import dynamic_embedding as de

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.keras import initializers as keras_init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import device_setter
from tensorflow.python.util import compat
import tensorflow as tf 

# tf.compat.v1.disable_eager_execution()
def _get_devices():
  return ["/gpu:0" if test_util.is_gpu_available() else "/cpu:0"]


default_config = config_pb2.ConfigProto(
    allow_soft_placement=False,
    gpu_options=config_pb2.GPUOptions(allow_growth=True))

# @test_util.deprecated_graph_mode_only
class WANGTENGLONG():
    def test_main(self):
        embeddings = de.get_variable(
                "t300",
                dtypes.int64,
                dtypes.float32,
                dim= 5,
                devices=_get_devices() * 2,
                initializer=2.0,
            )
        ids = constant_op.constant([0, 1, 2, 3, 4], dtype=dtypes.int64)
        embedding, trainable = de.embedding_lookup(embeddings,
                                                    ids,
                                                    max_norm=1.0,
                                                    return_trainable=True)
        with self.session(use_gpu=test_util.is_gpu_available(),
                            config=default_config):
            self.assertAllClose(
                embedding.eval(),
                [
                    [1.0],
                ] * 5,
            )

if __name__ == "__main__":
    w = WANGTENGLONG()
    w.test_main()
    # test.main()