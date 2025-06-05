using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEditor.U2D.Common;
using UnityEngine;

namespace UnityEditor.U2D.PSD
{
    internal class PSDImporterEditorExternalData : ScriptableObject
    {
        [SerializeField]
        public List<TextureImporterPlatformSettings> platformSettings = new List<TextureImporterPlatformSettings>();

        public void Init(PSDImporter importer, IList<TextureImporterPlatformSettings> platformSettingsNeeded)
        {
            var importerPlatformSettings = importer.GetAllPlatformSettings();
            
            for (int i = 0; i < importerPlatformSettings.Length; ++i)
            {
                var tip = importerPlatformSettings[i];
                var setting = platformSettings.FirstOrDefault(x => x.name == tip.name);
                if (setting == null)
                {
                    platformSettings.Add(tip);
                }
            }

            foreach (var ps in platformSettingsNeeded)
            {
                var setting = platformSettings.FirstOrDefault(x => x.name == ps.name);
                if (setting == null)
                {
                    platformSettings.Add(ps);
                }
            }
        }
    }    
}

