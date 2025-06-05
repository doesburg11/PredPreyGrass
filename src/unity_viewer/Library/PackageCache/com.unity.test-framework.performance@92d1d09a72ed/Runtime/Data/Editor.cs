using System;

namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents information about Editor version.
    /// </summary>
    [Serializable]
    public class Editor
    {
        /// <summary>
        /// Full editor version including changeset.
        /// </summary>
        [RequiredMember] public string Version;

        /// <summary>
        /// Branch name from which the editor was built.
        /// </summary>
        [RequiredMember] public string Branch;

        /// <summary>
        /// Changeset 12 characters long.
        /// </summary>
        [RequiredMember] public string Changeset;

        /// <summary>
        /// Editor version datetime in Unix Epoch milliseconds format.
        /// </summary>
        [RequiredMember] public int Date;
    }
}