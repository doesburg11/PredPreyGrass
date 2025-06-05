# Introduction to Aseprite Importer

The Aseprite Importer is an Asset importer that imports [Aseprite's](https://www.aseprite.org/) .ase and .aseprite files into Unity.

See [Aseprite Features](AsepriteFeatures) to read more about which features from Aseprite are supported by the importer. See [Importer Features](ImporterFeatures) to read more about which features the Aseprite Importer provides.

## Package versions

The different versions of the Aseprite Importer package are supported by the following versions of Unity respectively:

Package version  | Unity Editor version
--|--
1.2.x |  6.1
1.1.x  |  6.0, 2022.3 and 2021.3
1.0.x  |  6.0, 2022.3 and 2021.3

## Modifying generated Animator Controllers

The Aseprite Importer generates an Animator Controller if the Aseprite file contains more than one frame, the [Import Mode](ImporterFeatures#general) is set to **Animated Sprite**, and the **Animation Clip** checkbox is checked in the importer. This Animator Controller is Read-Only, meaning that it cannot be changed. To modify the Animator Controller, please see [the Importer FAQ](ImporterFAQ#how-to-make-changes-to-an-animator-controller).