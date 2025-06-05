using System;

using Codice.CM.Common;

namespace Unity.PlasticSCM.Editor.Views.History
{
    [Serializable]
    internal class SerializableHistoryTabState
    {
        internal RepositorySpec RepSpec { get; private set; }
        internal long ItemId { get; private set; }
        internal string Path { get; private set; }
        internal bool IsDirectory { get; private set; }

        internal bool IsInitialized { get; private set; }

        internal SerializableHistoryTabState(
            RepositorySpec repSpec,
            long itemId,
            string path,
            bool isDirectory)
        {
            RepSpec = repSpec;
            ItemId = itemId;
            Path = path;
            IsDirectory = isDirectory;

            IsInitialized = true;
        }
    }
}
