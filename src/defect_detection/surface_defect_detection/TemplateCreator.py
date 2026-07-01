#!/usr/bin/python3

import cv2
import numpy as np

from cv2.typing import MatLike
from enum import Enum
from typing import Optional

from src.common.image_loader import ImageLoader, ImageLoaderParameters
from src.common.image_saver import ImageSaverParameters
from src.common.launcher import Launcher, LaunchOption, LauncherParameters
from src.common.processors import ImageProcessor, DefaultKeysProcessor, KeyProcessor

from .config import *
from .TemplateMatcher import *

class Template:
    def __init__( self, mean:Optional[MatLike] = None, std:Optional[MatLike] = None ):
        self.mean = mean
        self.std = std

class TemplateNormalizer(ImageProcessor):
    def normalize( self, img:MatLike ) -> MatLike:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_norm = clahe.apply(l)

        lab_norm = cv2.merge([l_norm, a, b])
        img_norm = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)

        return img_norm
    
    def process_img(self, img: MatLike ) -> MatLike:
        return self.normalize( img )

class TemplateAccumulator:
    def __init__( self, template:Template ):
        self.__template = template
        self.__template_m2 = None
        self.__template_count = 0

    def add_template( self, img: MatLike ):
        if self.__template.mean is not None and img.shape != self.__template.mean.shape:
            return

        img = img.astype(np.float32)

        if self.__template.mean is None:
            self.__template.mean = img
            self.__template_m2 = np.zeros_like(img, dtype=np.float32)
            self.__template_count = 1
        else:
            self.__template_count += 1
            delta = img - self.__template.mean
            self.__template.mean += delta / self.__template_count
            delta2 = img - self.__template.mean
            self.__template_m2 += delta * delta2

        template_var = self.__template_m2 / self.__template_count
        self.__template.std = np.sqrt( template_var )

class TemplateCreator:
    def __init__( self, 
                  matcher:TemplateMatcher, 
                  filter:Optional[ImageProcessor] = None ):
        self.template = Template()
        self.matcher = matcher
        self.accumulator = TemplateAccumulator( self.template )
        self.filter = filter

    def reset( self ):
        self.template = Template()
        self.accumulator = TemplateAccumulator( self.template )

    def create_from_images( self, imgs:Sequence[MatLike] ):
        for img in imgs:
            preprocess = self.matcher.match( img )
            
            if ( self.filter is not None ):
                preprocess = self.filter.process_img( preprocess )

            self.accumulator.add_template( preprocess )

    def create_from_loader( self, loader: ImageLoader ):
        if loader.can_load():
            imgs = loader.load()
            self.create_from_images( imgs )

    def create_from_folder( self, folder_path, img_extension ):
        loader_params = ImageLoaderParameters( folder_path, img_extension, None )
        loader = ImageLoader( loader_params )
        self.create_from_loader( loader )

    def create_from_file( self, filepath ):
        loader = ImageLoader.from_filepath( filepath )
        if loader is not None:
            self.create_from_loader( loader )
        
    
class TemplateCreatorProcessor( ImageProcessor, DefaultKeysProcessor ):
    class ActionType(Enum):
        NONE = 0
        ADD = 1
        ADD_ALL = 2
        CLEAR = 3

    class DisplayType(Enum):
        MEAN = 0
        STD = 1

    def __init__( self,
                template_img,
                template_loader_params:ImageLoaderParameters,
                img_saver_params:ImageSaverParameters, 
                process_img_saver_params:ImageSaverParameters ):
        DefaultKeysProcessor.__init__( self, img_saver_params, process_img_saver_params )
        self.__template_loader = ImageLoader( template_loader_params )
        
        matcher = MinAreaRectShapeMatcher( template_img )
        normalizer = TemplateNormalizer()
        
        self.__creator = TemplateCreator( matcher, normalizer )
        self.__display_type = self.DisplayType.MEAN
        self.__action_type = self.ActionType.ADD
        self.sub_menus().update( {
            'm' : KeyProcessor( 'm', "Set display type to mean", lambda img, process: self.__set_display_type( self.DisplayType.MEAN ) ),
            'd' : KeyProcessor( 'd', "Set display type to std", lambda img, process: self.__set_display_type( self.DisplayType.STD ) ),
            '+' : KeyProcessor( '+', "Add image to template creator", lambda img, process: self.__set_action_type( self.ActionType.ADD ) ),
            'a' : KeyProcessor( 'a', "Add all images to template creator", lambda img, process: self.__set_action_type( self.ActionType.ADD_ALL ) ),
            'c' : KeyProcessor( 'c', "Clear template creator", lambda img, process: self.__set_action_type( self.ActionType.CLEAR ) ),
        } )

    def __set_display_type( self, type ):
        self.__display_type = type
        print( f"Set display type {self.__display_type.name}")

    def __set_action_type( self, type ):
        self.__action_type = type
        print( f"Set action type {self.__action_type.name}")

    def process_img( self, img:MatLike ) -> MatLike:
        if self.__action_type == self.ActionType.ADD:
            self.__creator.create_from_images( [img.copy()] )
        elif self.__action_type == self.ActionType.ADD_ALL:
            self.__creator.create_from_loader( self.__template_loader )
        elif self.__action_type == self.ActionType.CLEAR:
            self.__creator.reset()

        self.__set_action_type( self.ActionType.NONE )

        if self.__creator.template.mean is None:
            return img.copy()
        
        if self.__display_type == self.DisplayType.STD:
            return self.__creator.template.std.astype( np.uint8 ) 
        
        return self.__creator.template.mean.astype( np.uint8 )
    
    def title(self) -> str:
        return "Template Creator processor"

def main():
    img_loader_params = ImageLoaderParameters( DATA_GOOD_PATH, IMAGE_EXTENSION, None )
    img_saver_params = ImageSaverParameters( DATA_IGNORED_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )
    process_img_saver_params = ImageSaverParameters( DATA_IGNORED_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )

    template_img_loader_params = ImageLoaderParameters( DATA_GOOD_PATH, IMAGE_EXTENSION, GOOD_IMAGE_BASE_NAME )
    template_img_loader = ImageLoader( template_img_loader_params )
    template_img = template_img_loader.load()[0]

    matcher_processor = TemplateCreatorProcessor( template_img, img_loader_params, img_saver_params, process_img_saver_params )
    
    launcher_params = LauncherParameters( 
        img_loader_params,
        img_saver_params,
        process_img_saver_params,
        0, 
        LaunchOption.LOAD_IMAGE,
        SHOW_IMAGE )
    
    launcher = Launcher( 
        launcher_params, 
        matcher_processor, 
        matcher_processor )
    
    launcher.launch()

if ( __name__ == "__main__" ):
    main()