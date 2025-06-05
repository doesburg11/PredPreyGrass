#if ENABLE_URP_14_0_0_OR_NEWER
using System;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using U2DPackage = UnityEngine.U2D;

#if UNITY_6000_0_OR_NEWER
using URPPackage = UnityEngine.Rendering.Universal;
#else
using URPPackage = UnityEngine.Experimental.Rendering.Universal;
#endif //UNITY_6000_0_OR_NEWER


namespace UnityEditor.Rendering.Universal
{
    internal sealed class U2DToURPPixelPerfectConverter : RenderPipelineConverter
    {
        public override string name => "2D to URP Pixel Perfect Camera Converter";
        public override string info => "This will upgrade all 2D Pixel Perfect Camera to the URP version.";
        public override int priority => -2000;
        public override Type container => typeof(UpgradeURP2DAssetsContainer);

        List<string> m_AssetsToConvert = new List<string>();

        public static bool UpgradePixelPerfectCamera(U2DPackage.PixelPerfectCamera cam)
        {
            if (cam == null)
                return false;

            // Copy over serialized data
            var urpCam = cam.gameObject.AddComponent<URPPackage.PixelPerfectCamera>();

            if (urpCam == null)
                return false;

            urpCam.assetsPPU = cam.assetsPPU;
            urpCam.refResolutionX = cam.refResolutionX;
            urpCam.refResolutionY = cam.refResolutionY;

            if (cam.upscaleRT)
                urpCam.gridSnapping = URPPackage.PixelPerfectCamera.GridSnapping.UpscaleRenderTexture;
            else if(cam.pixelSnapping)
                urpCam.gridSnapping = URPPackage.PixelPerfectCamera.GridSnapping.PixelSnapping;

            if (cam.cropFrameX && cam.cropFrameY)
            {
                if (cam.stretchFill)
                    urpCam.cropFrame = URPPackage.PixelPerfectCamera.CropFrame.StretchFill;
                else
                    urpCam.cropFrame = URPPackage.PixelPerfectCamera.CropFrame.Windowbox;
            }
            else if (cam.cropFrameX)
            {
                urpCam.cropFrame = URPPackage.PixelPerfectCamera.CropFrame.Pillarbox;
            }
            else if (cam.cropFrameY)
            {
                urpCam.cropFrame = URPPackage.PixelPerfectCamera.CropFrame.Letterbox;
            }
            else
            {
                urpCam.cropFrame = URPPackage.PixelPerfectCamera.CropFrame.None;
            }

            UnityEngine.Object.DestroyImmediate(cam, true);

            EditorUtility.SetDirty(urpCam);

            return true;
        }

        void UpgradeGameObject(GameObject go)
        {
            var cam = go.GetComponentInChildren<U2DPackage.PixelPerfectCamera>();

            if(cam != null)
                UpgradePixelPerfectCamera(cam);
        }

        public override void OnInitialize(InitializeConverterContext context, Action callback)
        {
            string[] allAssetPaths = AssetDatabase.GetAllAssetPaths();

            foreach (string path in allAssetPaths)
            {
                if (URP2DConverterUtility.IsPrefabOrScenePath(path, "m_AssetsPPU:"))
                {
                    ConverterItemDescriptor desc = new ConverterItemDescriptor()
                    {
                        name = Path.GetFileNameWithoutExtension(path),
                        info = path,
                        warningMessage = String.Empty,
                        helpLink = String.Empty
                    };

                    // Each converter needs to add this info using this API.
                    m_AssetsToConvert.Add(path);
                    context.AddAssetToConvert(desc);
                }
            }

            callback.Invoke();
        }

        public override void OnRun(ref RunItemContext context)
        {
            string result = string.Empty;
            string ext = Path.GetExtension(context.item.descriptor.info);

            if (ext == ".prefab")
                result = URP2DConverterUtility.UpgradePrefab(context.item.descriptor.info, UpgradeGameObject);
            else if (ext == ".unity")
                URP2DConverterUtility.UpgradeScene(context.item.descriptor.info, UpgradeGameObject);

            if (result != string.Empty)
            {
                context.didFail = true;
                context.info = result;
            }
            else
            {
                context.hasConverted = true;
            }
        }

        public override void OnClicked(int index)
        {
            EditorGUIUtility.PingObject(AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(m_AssetsToConvert[index]));
        }

        public override void OnPostRun()
        {
            Resources.UnloadUnusedAssets();
        }
    }
}
#endif