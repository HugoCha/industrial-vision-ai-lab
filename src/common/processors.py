#!/usr/bin/python3

from abc import ABC, abstractmethod
from cv2.typing import MatLike
from enum import Enum
from typing import Callable, Optional

from src.common.image_saver import ImageSaver, ImageSaverParameters

class ProcessKeyResult(Enum):
    Continue = 0
    Quit = 1
    Next = 2
    Previous = 3 

class ImageProcessor(ABC):
    @abstractmethod
    def process_img( self, img:MatLike ) -> MatLike:
        pass

class KeyProcessor:
    def __init__( self, key:str, description:str, func:Callable[[Optional[MatLike], Optional[MatLike]], None] ):
        self.key = key
        self.description = description
        self.func = func

class KeysProcessor(ABC):
    def process_key( self, key:int, img:MatLike, process:Optional[MatLike] ) -> None:
        if ( key < 0 ): return None
        if ( chr( key ) == 'h' ):
            print( self.menu() )
            return None

        if ( chr( key ) in self.sub_menus() ):
            self.sub_menus()[chr( key )].func( img, process )
        else:
            print( f"Unknown key: {chr(key)}" )

    @abstractmethod
    def title( self ) -> str:
        pass
    
    @abstractmethod
    def sub_menus( self ) -> dict[str,KeyProcessor]:
        pass

    def previous_menu( self ) -> str:
        return "Previous"
    
    def next_menu( self ) -> str:
        return "Next"

    def quit_menu( self ) -> str:
        return "Quit"

    def menu( self ) -> str:
        menu = []
        menu.append( f"================================\n" )
        menu.append( f"{self.title()} menu:\n" )
        menu.append( f"================================\n" )
        menu.append( f"'h': Display menu\n" )

        sub_menus = self.sub_menus()
        if sub_menus is not None:
            for key in sub_menus:
                menu.append( f"'{key}' : {sub_menus[key].description}\n")
        menu.append( f"'◀': {self.previous_menu()}\n" )
        menu.append( f"'▶': {self.next_menu()}\n" )
        menu.append( f"'q': {self.quit_menu()}\n" )
        return "".join( menu )

class DefaultImageProcessor(ImageProcessor):
    def process_img( self, img:MatLike ) -> MatLike:
        return img
    
class DefaultKeysProcessor(KeysProcessor):
    def __init__( self, 
                  img_saver_params:ImageSaverParameters,
                  process_img_saver_params:ImageSaverParameters, ):
        self.__raw_img_saver = ImageSaver( img_saver_params )
        self.__process_img_saver = ImageSaver( process_img_saver_params )
        
        self.__sub_menus:dict[str, KeyProcessor] = {
            'o': KeyProcessor( 'o', f"Save original image in :{self.__raw_img_saver.get_dirpath()}", lambda im, proc : self.__raw_img_saver.save( im ) ),
            'm': KeyProcessor( 'm', f"Save modified image in :{self.__process_img_saver.get_dirpath()}", lambda im, proc : self.__process_img_saver.save( proc ) ),
            'b': KeyProcessor( 'b', f"Save original and modified image in :{self.__raw_img_saver.get_dirpath()}, {self.__process_img_saver.get_dirpath()}", lambda im, proc : self.__raw_img_saver.save( im ) and self.__process_img_saver.save( proc ) )
        }

    def title( self ) -> str:
        return "Default menu"
    
    @property
    def raw_image_saver( self ):
        return self.__raw_img_saver
    
    @property
    def processed_image_saver( self ):
        return self.__process_img_saver
    
    def sub_menus(self) -> dict[str, KeyProcessor]:
        return self.__sub_menus