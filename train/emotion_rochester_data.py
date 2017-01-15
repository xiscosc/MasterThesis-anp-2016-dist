# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Small library that points to the Emotion Rochester data set.
"""
from __future__ import absolute_import

from train.dataset import Dataset


class EmotionRochesterData(Dataset):
    """
    Emotion Rochester dataset
    """

    def __init__(self, subset):
        super(EmotionRochesterData, self).__init__('EmotionRochester', subset)

    def num_classes(self):
        """Returns the number of classes in the data set."""
        return 8

    def num_examples_per_epoch(self):
        """Returns the number of examples in the data set."""
        if self.subset == 'train':
            return 18508
        if self.subset == 'val':
            return 4631

    def download_message(self):
        pass
