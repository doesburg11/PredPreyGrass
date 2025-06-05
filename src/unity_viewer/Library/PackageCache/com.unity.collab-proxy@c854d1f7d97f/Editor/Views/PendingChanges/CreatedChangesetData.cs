using Codice.CM.Common;

namespace Unity.PlasticSCM.Editor.Views.PendingChanges
{
    internal class CreatedChangesetData
    {
        internal enum Type
        {
            Checkin,
            Shelve
        }

        internal Type OperationType { get; private set; }
        internal long CreatedChangesetId { get; private set; }
        internal RepositorySpec RepositorySpec { get; private set; }

        internal CreatedChangesetData(
            Type operationType,
            long createdChangesetId,
            RepositorySpec repositorySpec)
        {
            OperationType = operationType;
            CreatedChangesetId = createdChangesetId;
            RepositorySpec = repositorySpec;
        }
    }
}
