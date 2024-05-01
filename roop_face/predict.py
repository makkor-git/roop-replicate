import os
import sys
import shutil
import zipfile
import urllib.request
from argparse import Namespace
from roop import core as r
from cog import BasePredictor, Input, Path as CogPath
import asyncio

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        pass

    def predict(self,
        source_image: CogPath = Input(
            description="Source image"), 
        target_gif: CogPath = Input(
            description="Target gif")
    ) -> CogPath:
        file = r.run_replicate(source_image, target_gif)
        print(f"[+] Gif generated at {file}")
        return CogPath(file)