# Introduction to PSD Importer
The **PSD Importer** is an [Asset importer](https://docs.unity3d.com/ScriptReference/AssetImporter.html) that imports [Adobe Photoshop .psb](https://helpx.adobe.com/photoshop/using/file-formats.html#large_document_format_psb) files into Unity, and generates a [Prefab](https://docs.unity3d.com/Manual/Prefabs.html) of Sprites based on the imported source file. The .psb file format is functionally identical to the more common Adobe [.psd format](https://helpx.adobe.com/photoshop/using/file-formats.html#photoshop_format_psd), but supports much larger images than the .psd format (up to 300,000 pixels in any dimension). To convert existing artwork from .psd to .psb format, you can open them in Adobe Photoshop and then save them as .psb files.

Importing .psb files with the PSD Importer allows you to use features such as [Mosaic](PSD-importer-properties.md#Mosaic) (to automatically generate a Sprite sheet from the imported layers) and [Character Rig](PSD-importer-properties.md#Rig) (to reassemble the Sprites of a character as they are arranged in their source files).

The PSD Importer currently only supports two [Texture Modes](https://docs.unity3d.com/Manual/TextureTypes.html):[ Default](https://docs.unity3d.com/Manual/TextureTypes.html#Default) and[ Sprite](https://docs.unity3d.com/Manual/TextureTypes.html#Sprite).

The different versions of the PSD Importer package are supported by the following versions of Unity respectively:

Package version  | Unity Editor version
--|--
10.x.x |  6000.1
9.x.x  |  2023.1 up to 6000.0
8.x.x  |  2022.2
7.x.x  |  2022.1
6.x.x  |  2021.2 or 2021.3
5.x.x  |  2020.2 or 2020.3

## Support for .psd files

By default .psd files are imported with the Texture Importer. If you wish to instead import a .psd file with the PSD Importer, simply select the .psd file, click on the Importer dropdown and select **UnityEditor.U2D.PSD.PSDImporter**.

![](images/psd-file-import.png)<br/>Importer drop down.

For information on how to automate the selection of an importer, see the [PSD File Importer Override](PSD-override.md) section.

## Supported and unsupported Photoshop effects

When importing a .psb file into Unity with the PSD Importer, the importer generates a prefab of Sprites based on the image and layer data of the imported .psb file. To ensure the importer imports the file correctly, ensure that the Photoshop file is saved with [Maximize Compatibility](https://helpx.adobe.com/au/photoshop/using/file-formats.html#maximize_compatibility_for_psd_and_psb_files) enabled. 

The PSD Importer does not support all of Photoshop’s layer and visual effects or features. The PSD Importer ignores the following Photoshop layer properties and visual effects when it generates the Sprites and prefab:

- Channels
- Blend Modes
- Layer Opacity
- Effects

If you want to add visual effects to the generated Sprites, you can add additional Textures to the Sprites with the [Sprite Editor's Secondary Textures](https://docs.unity3d.com/Manual/SpriteEditor-SecondaryTextures.html) module. Shaders can sample these Secondary Textures to apply additional effects to the Sprite, such as normal mapping. Refer to the [Sprite Editor: Secondary Textures](https://docs.unity3d.com/Manual/SpriteEditor-SecondaryTextures.html) documentation for more information.
