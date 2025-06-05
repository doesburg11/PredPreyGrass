using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal class AsepriteImporterEditorExternalData : ScriptableObject
    {
        [SerializeField]
        public List<TextureImporterPlatformSettings> platformSettings = new();

        public void Init(AsepriteImporter importer, IList<TextureImporterPlatformSettings> platformSettingsNeeded)
        {
            var importerPlatformSettings = importer.GetAllPlatformSettings();

            foreach (var tip in importerPlatformSettings)
            {
                var j = 0;
                for (j = 0; j < platformSettings.Count; ++j)
                {
                    if (platformSettings[j].name == tip.name)
                        break;
                }

                if (j >= platformSettings.Count)
                    platformSettings.Add(tip);
            }

            foreach (var ps in platformSettingsNeeded)
            {
                var j = 0;
                for (j = 0; j < platformSettings.Count; ++j)
                {
                    if (platformSettings[j].name == ps.name)
                        break;
                }

                if (j >= platformSettings.Count)
                    platformSettings.Add(ps);
            }
        }
    }
}
