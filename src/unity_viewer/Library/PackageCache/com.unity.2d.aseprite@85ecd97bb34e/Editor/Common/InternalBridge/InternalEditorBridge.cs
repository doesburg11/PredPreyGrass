using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.U2D;

namespace UnityEditor.U2D.Aseprite.Common
{
    internal static class InternalEditorBridge
    {
        public static bool DoesHardwareSupportsFullNPOT()
        {
            return ShaderUtil.hardwareSupportsFullNPOT;
        }

        public static Texture2D CreateTemporaryDuplicate(Texture2D tex, int width, int height)
        {
            return SpriteUtility.CreateTemporaryDuplicate(tex, width, height);
        }

        public static void ShowSpriteEditorWindow(Object obj = null)
        {
            SpriteUtilityWindow.ShowSpriteEditorWindow(obj);
        }

        public static void ApplySpriteEditorWindow()
        {
            SpriteUtilityWindow.ApplySpriteEditorWindow();
        }

        public static void AddManagedGameObject(this PreviewRenderUtility scene, GameObject go) => scene.AddManagedGO(go);

        public static void RefreshInspectors() => InspectorWindow.RefreshInspectors();

        public static void GenerateOutlineFromSprite(Sprite sprite, float detail, byte alphaTolerance, bool holeDetection, out Vector2[][] paths)
        {
            UnityEditor.Sprites.SpriteUtility.GenerateOutlineFromSprite(sprite, detail, alphaTolerance, holeDetection, out paths);
        }

        public static void SetSpriteAtlasToV2(SpriteAtlas atlas) => atlas.SetV2();

        public static void RegisterAndPackSpriteAtlas(SpriteAtlas atlas, AssetImportContext ctx, AssetImporter importer, ScriptablePacker packer)
        {
            atlas.RegisterAndPackAtlas(ctx, importer, packer);
        }
    }
}
