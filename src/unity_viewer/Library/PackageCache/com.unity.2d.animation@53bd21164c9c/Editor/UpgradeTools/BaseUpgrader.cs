using System.Collections.Generic;
using UnityEngine;

namespace UnityEditor.U2D.Animation.Upgrading
{
    internal abstract class BaseUpgrader
    {
        protected Logger m_Logger = new Logger();

        internal abstract List<Object> GetUpgradableAssets();
        internal abstract UpgradeReport UpgradeSelection(List<ObjectIndexPair> selection);
    }
}
