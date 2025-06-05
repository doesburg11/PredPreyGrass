using System.Collections.Generic;

namespace UnityEngine.U2D.IK
{
    internal class AlwaysUpdateCullingStrategy : BaseCullingStrategy
    {
        public override bool AreBonesVisible(IList<int> transformIds) => true;
    }
}
