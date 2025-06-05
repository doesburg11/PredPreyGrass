# Introduction to 2D Animation
The 2D Animation package includes features and tools that allow you to quickly rig and animate 2D characters in Unity in a variety of ways. The different versions of the 2D Animation package are supported by the following versions of Unity respectively:

Package version  | Unity Editor version
--|--
10.x.x |  2023.1
9.x.x  |  2022.2
8.x.x  |  2022.1
7.x.x  |  2021.2 or 2021.3
5.x.x  |  2020.2 or 2020.3

## 2D Animation and PSD Importer package integration
Use the 2D Animation package with the [PSD Importer](https://docs.unity3d.com/Packages/com.unity.2d.psdimporter@latest) package to easily import character artwork created in Photoshop into Unity, and prepare it for animation with the 2D Animation package. The PSD Importer is an [Asset importer](https://docs.unity3d.com/Manual/ImportingAssets.html) that supports Adobe Photoshop .psb files, and generates a Prefab made of Sprites based on the source file and its [layers](https://helpx.adobe.com/photoshop/using/layer-basics.html). The generated Prefab of a character or prop to be animated with the 2D Animation package is called an 'actor'.

The [.psb](https://helpx.adobe.com/photoshop/using/file-formats.html#large_document_format_psb)[ file format](https://helpx.adobe.com/photoshop/using/file-formats.html#large_document_format_psb) has identical functions as the more common Adobe .psd format, with additional support for much larger image sizes. Refer to the [PSD Importer](https://docs.unity3d.com/Packages/com.unity.2d.psdimporter@latest) package documentation for more information about the importer’s features.

## Upgrading Sprite Library Assets and Animation Clips
The 2D Animation package and its assets are often updated with major and minor tweaks over time. Some asset improvements can be automatically applied when you upgrade to the latest version of the package. However, some of these changes require a manual step in order to have the assets use the latest code path.

The 2D Animation Asset Upgrader tool eases the transition and upgrade of older assets to newer ones. Refer to the [2D Animation Asset Upgrader](AssetUpgrader.md) section for more information.
