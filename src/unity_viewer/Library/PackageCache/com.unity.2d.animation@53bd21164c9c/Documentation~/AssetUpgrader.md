# 2D Animation Asset Upgrader
The 2D Animation package and its assets are often updated with major and minor tweaks over time. Some asset improvements can be automatically applied when you upgrade to the latest version of the package. However, some of these changes require a manual step in order to have the assets use the latest code path.

The 2D Animation Asset Upgrader tool eases the transition and upgrade of older assets to newer ones. This tool has the following features:

- Upgrades [Sprite Library Asset](SL-Asset.md) files ending in `.asset` to Sprite Library Source Asset files ending in `.spriteLib`.
- Moves Sprite Library Assets baked into `.psb` files created in Unity 2019 and Unity 2020 out into their own separate Sprite Library Source Asset files ending in `.spriteLib`.
- Upgrades [Animation Clips](https://docs.unity3d.com/Manual/AnimationClips.html) that animate Sprites based on the [Sprite Resolver component](https://docs.unity3d.com/Packages/com.unity.2d.animation@latest/index.html?subfolder=/manual/SL-Resolver.html)'s [Category and Label](SL-Editor.md) hash in Unity 2019 and Unity 2020, to Sprite Resolver's new **Sprite Hash** property from Unity 2022 onwards.
- Upgrades Animation Clips animating the Sprite Resolver component's **Sprite Key** property in Unity 2021, to Sprite Resolver's new Sprite Hash property from Unity 2022 onwards.

## Getting Started
Before upgrading any assets in your current project, make sure to source control or back up your project.

Go to **Window > 2D > 2D Animation Asset Upgrader** to open the 2D Animation Asset Upgrader.
<br/>![](images/AssetUpgrader_Window.png)

## Upgrading Sprite Libraries
Follow these steps to upgrade the Sprite Libraries in the project.
1. [Open](#getting-started) the 2D Animation Asset Upgrader.

2. Select the __Sprite Libraries__ button to open the Sprite Libraries tab.

3. Select the __Scan project__ button. The window then displays a list of all the Sprite Libraries that you can upgrade.
![](images/AssetUpgrader_SpriteLibToUpgrade.png)

4. Clear any Sprite Library Assets which you do not want to upgrade.

5. Select the __Upgrade selected__ button to begin the upgrading process.

6. The editor then displays a pop-up window to inform you that the upgrade cannot be reverted, and any [Asset Bundles](https://docs.unity3d.com/Manual/AssetBundlesIntro.html) connected to the Sprite Libraries will need to be rebuilt. Select __Yes__ to proceed with the upgrade, or __No__ to cancel the upgrading process.

7. Once the upgrading process is complete, the 2D Animation Asset Upgrader will display the status of the upgrading process for each of the selected assets.
![](images/AssetUpgrader_SpriteLibUpgraded.png)

8. Select the __Open upgrade log__ button to get more detailed information about the different upgrade warnings and errors that may appear. The upgrade log will also list all the Asset Bundles that need to be rebuilt for the upgrading process.

## Upgrade Animation Clips
Follow these steps to upgrade the Animation Clips in the project:
1. Open the 2D Animation Asset Upgrader.

2. Select the **Animation Clips** button to open the Animation Clips tab.

3. Select the **Scan project** button. The window then displays a list of all the Animation Clips that you can upgrade.
![](images/AssetUpgrader_AnimationClipsToUpgrade.png)

4. Clear any Animation Clips which you do not want to upgrade.

5. Select the __Upgrade selected__ button to begin the upgrading process.

6. The editor then displays a pop-up window to inform you that the upgrade cannot be reverted, and that any Asset Bundles connected to the Animation Clips will need to be rebuilt. Select __Yes__ to proceed with the upgrade, or __No__ to cancel the upgrading process.

7. Once the upgrading process is complete, the 2D Animation Asset Upgrader will display the status the upgrading process for each of the selected Animation Clips.
![](images/AssetUpgrader_AnimationClipsUpgraded.png)

8. Select the __Open upgrade log__ button to get more detailed information about the upgrade warnings and errors that may appear. The upgrade log will also list all the Asset Bundles that need to be rebuilt for the upgrading process.

## Common upgrading errors and solutions
The following are the common errors you may face when upgrading your projects, and their suggested solutions.

### Referring to the upgrade log for general debugging
The first step to debugging an upgrade failure is to refer to the upgrade log. The 2D Animation Asset Upgrader writes each action it takes into the log, helping you to track down the point of failure.

```
9/11/2021 2:06:23 PM
Asset Upgrading
---------------
Verifying if the asset Talk (UnityEngine.AnimationClip) is an AnimationClip.
Extracting SpriteHash bindings from clip. Found 0 bindings.
Extracting SpriteKey bindings from clip. Found 0 bindings.
Extracting Category bindings from clip. Found 0 bindings.
Extracting Label keyframes from clip. Found 3 keyframes.
Extracting Label bindings from clip. Found 1 bindings.
Merging different types keyframes from the same bindings, into the same binding list. We now have 1 binding lists.
Order the keyframe data in binding=Hand by time.
Converting keyframes into uniformed format for binding=Hand
Cannot find a category for keyframe at time=0 in binding=Hand.
Cannot find a category for keyframe at time=3.383333 in binding=Hand.
Cannot find a category for keyframe at time=4.966667 in binding=Hand.
Expected 3 keyframes after cleanup for binding=Hand, but ended up with 0.
The upgrade of the clip Talk failed. Some keyframes could not be converted in the animation clip.
```
In this example, the upgrade log shows the actions and results from upgrading an Animation Clip that contains only Label hashes, but no Category hashes. The 2D Animation Upgrader writes to the log that it cannot find a Category hash for three out of three keyframes. This results in a failure to upgrade the Animation Clip.

### When some keyframes could not be converted in the Animation Clip
![](images/AssetUpgrader_Err_CouldNotBeConverted.png)
One of the most common reasons for this error is when an Animation Clip contains either Sprite Resolver Category hash keys or Sprite Resolver Label hash keys, but not both types of hash keys at the same time.

![](images/AssetUpgrader_Err_CatLabel_InCorrect.png)
This example shows an incorrect Animation Clip setup where the Animation Clip contains only the Label hash, but not a Category hash, which leads to the above error.

![](images/AssetUpgrader_Err_CatLabel_Correct.png)
This example shows the corrected setup, where the Animation Clip contains both the Label hash and the Category hash.

Fix this error by recording a Sprite Swap on the first frame in the Animation Clip. Once the Sprite Swap is added, the 2D Animation Asset Upgrader is able to upgrade the Animation Clip.
