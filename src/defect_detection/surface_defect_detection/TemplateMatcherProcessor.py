#!/usr/bin/python3

from src.common.image_loader import ImageLoaderParameters
from src.common.image_saver import ImageSaverParameters
from src.common.launcher import Launcher, LaunchOption, LauncherParameters
from src.common.processors import ImageProcessor, DefaultKeysProcessor, KeyProcessor
from src.common.vision_utils import *
from src.common.visualization import *

from .config import *
from .TemplateMatcher import *

class TemplateMatcherProcessor( ImageProcessor, DefaultKeysProcessor ):
    def __init__( self,
                  template_img: MatLike,
                  img_saver_params:ImageSaverParameters, 
                  process_img_saver_params:ImageSaverParameters ):
        DefaultKeysProcessor.__init__( self, img_saver_params, process_img_saver_params )
        self.template_img = template_img
        self.matcher = ContourFitShapeMatcher( template_img )
        self.sub_menus().update( {
            'c' : KeyProcessor( 'c', "Use contour Shape Matcher", lambda img, process: self._set_matcher( ContourFitShapeMatcher ) ),
            'e' : KeyProcessor( 'e', "Use ECC Shape Matcher", lambda img, process: self._set_matcher( ECCShapeMatcher ) ),
            'o' : KeyProcessor( 'o', "Use ORB Shape Matcher", lambda img, process: self._set_matcher( ORBShapeMatcher ) ),
            'm' : KeyProcessor( 'm', "Use Min area rect Shape Matcher", lambda img, process: self._set_matcher( MinAreaRectShapeMatcher ) ),
            'p' : KeyProcessor( 'p', "Use PCA Shape Matcher", lambda img, process: self._set_matcher( PCAShapeMatcher ) ),
        } )

    def _set_matcher( self, matcher_cls ):
        self.matcher = matcher_cls( self.template_img )

    def process_img( self, img:MatLike ) -> MatLike:
        return self.matcher.match( img )
    
    def title(self) -> str:
        return "Shape matcher processor"

def main():
    img_loader_params = ImageLoaderParameters( DATA_BAD_PATH, IMAGE_EXTENSION, None )
    img_saver_params = ImageSaverParameters( DATA_BAD_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )
    process_img_saver_params = ImageSaverParameters( DATA_IGNORED_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )
    
    template_img_loader_params = ImageLoaderParameters( DATA_GOOD_PATH, IMAGE_EXTENSION, GOOD_IMAGE_BASE_NAME )
    template_img_loader = ImageLoader( template_img_loader_params )
    template_img = template_img_loader.load()[0]

    matcher_processor = TemplateMatcherProcessor( template_img, img_saver_params, process_img_saver_params )
    
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