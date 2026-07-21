#!/usr/bin/python3

from src.common.ColorNormalizer import *
from src.common.image_loader import ImageLoaderParameters
from src.common.image_saver import ImageSaverParameters
from src.common.launcher import Launcher, LaunchOption, LauncherParameters
from src.common.processors import ImageProcessor, DefaultKeysProcessor, KeyProcessor
from src.common.vision_utils import *
from src.common.visualization import *

from .config import *
from .Inspector import *
from .TemplateMatcher import *

class SurfaceDefectProcessor( ImageProcessor, DefaultKeysProcessor ):
    def __init__( self,
                  template_img,
                  img_saver_params:ImageSaverParameters, 
                  process_img_saver_params:ImageSaverParameters ):
        
        DefaultKeysProcessor.__init__( self, img_saver_params, process_img_saver_params )
        self.template_img = template_img
        self.matcher = MinAreaRectShapeMatcher( template_img )
        
        self.template_creator_HSV = TemplateCreator( MinAreaRectShapeMatcher( template_img ), ColorNormalizer( ColorSpace.HSV ) )
        self.template_creator_HSV.create_from_folder( DATA_GOOD_PATH, IMAGE_EXTENSION )
        self.color_inspector = ColorDefectInspector( self.template_creator_HSV )
        
        self.template_creator_GRAY = TemplateCreator( MinAreaRectShapeMatcher( template_img ), ColorNormalizer( ColorSpace.GRAY ) )
        self.template_creator_GRAY.create_from_folder( DATA_GOOD_PATH, IMAGE_EXTENSION )
        self.material_inspector = MaterialDefectInspector( self.template_creator_GRAY )

        self.global_inspector = MultipleDefectInspector( [self.color_inspector, self.material_inspector] )
        
        self.inspector_type = InspectorType.ALL

        self.sub_menus().update( {
            'a' : KeyProcessor( 'a', "Use all defect inspectors", lambda img, process: self._set_inspector( InspectorType.ALL ) ),
            'c' : KeyProcessor( 'c', "Use color defect inspector", lambda img, process: self._set_inspector( InspectorType.COLOR ) ),
            'm' : KeyProcessor( 'm', "Use material defect inspector", lambda img, process: self._set_inspector( InspectorType.MATERIAL ) ),
        } )

    def _set_inspector( self, type ):
        self.inspector_type = type

    def process_img( self, img:MatLike ) -> MatLike:
        affine_img = self.matcher.match( img )
        
        if self.inspector_type == InspectorType.COLOR:
            defect = self.color_inspector.inspect( affine_img )
        elif self.inspector_type == InspectorType.MATERIAL:
            defect = self.material_inspector.inspect( affine_img )
        else:
            defect = self.global_inspector.inspect( affine_img )

        process = self.draw_defect( affine_img, defect )
        return process
    
    def draw_defect( self, img:MatLike, defect:Defect, color=(0,255,0), thickness=1 ) -> MatLike:
        return cv2.drawContours( img, defect.contours, -1, color, thickness )
    
    def title(self) -> str:
        return "Surface defection processor"

def main():
    img_loader_params = ImageLoaderParameters( DATA_BAD_PATH, IMAGE_EXTENSION, IMAGE_LOAD_NAME )
    #img_loader_params = ImageLoaderParameters( DATA_GOOD_PATH, IMAGE_EXTENSION, IMAGE_LOAD_NAME )
    img_saver_params = ImageSaverParameters( DATA_IGNORED_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )
    process_img_saver_params = ImageSaverParameters( DATA_IGNORED_PATH, "", IMAGE_EXTENSION, CAN_OVERRIDE )
    
    template_img_loader_params = ImageLoaderParameters( DATA_GOOD_PATH, IMAGE_EXTENSION, GOOD_IMAGE_BASE_NAME )
    template_img_loader = ImageLoader( template_img_loader_params )
    template_img = template_img_loader.load()[0]
    processor = SurfaceDefectProcessor( template_img, img_saver_params, process_img_saver_params )

    launcher_params = LauncherParameters( 
        img_loader_params,
        img_saver_params,
        process_img_saver_params,
        0, 
        LaunchOption.LOAD_IMAGE,
        SHOW_IMAGE )
    
    launcher = Launcher( 
        launcher_params, 
        processor, 
        processor )
    
    launcher.launch()

if ( __name__ == "__main__" ):
    main()