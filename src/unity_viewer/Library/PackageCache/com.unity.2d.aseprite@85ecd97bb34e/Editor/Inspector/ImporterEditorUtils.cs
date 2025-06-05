using UnityEngine.UIElements;

namespace UnityEditor.U2D.Aseprite
{
    internal static class ImporterEditorUtils
    {
        const string k_DarkSkinUssClass = "asepriteImporter-editor-dark";
        const string k_LightSkinUssClass = "asepriteImporter-editor-light";

        public static void AddSkinUssClass(VisualElement element)
        {
            if (EditorGUIUtility.isProSkin)
                element.AddToClassList(k_DarkSkinUssClass);
            else
                element.AddToClassList(k_LightSkinUssClass);
        }
    }
}
