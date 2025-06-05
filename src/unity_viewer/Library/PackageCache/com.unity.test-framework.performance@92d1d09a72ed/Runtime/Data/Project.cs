namespace Unity.PerformanceTesting.Data
{
    /// <summary>
    /// Represents Project version.
    /// </summary>
    public class Project
    {
        /// <summary>
        /// Name of the project.
        /// </summary>
        [RequiredMember] public string Name;

        /// <summary>
        /// Version of the project unrelated to changeset e.g. package version.
        /// </summary>
        [RequiredMember] public string Version;

        /// <summary>
        /// Commit branch.
        /// </summary>
        [RequiredMember] public string Branch;

        /// <summary>
        /// Commit changeset.
        /// </summary>
        [RequiredMember] public string Changeset;

        /// <summary>
        /// Commit datetime in Unix Epoch milliseconds format.
        /// </summary>
        [RequiredMember] public int Date;
    }
}
