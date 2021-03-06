#
# Module manifest for module 'Posh-IC'
#
# Source https://raw.githubusercontent.com/gildas/posh-ic/master/Posh-IC.psd1
# Generated by: gildas.cherruel
# Generated on: 1/14/2016
#

@{

# Script module or binary module file associated with this manifest.
# RootModule = ''
RootModule = 'Posh-IC.psm1'

# Version number of this module.
ModuleVersion = '0.0.5'

# ID used to uniquely identify this module
GUID = '1fa1995f-9700-45ec-bd3a-db26f5aaf91e'

# Project Uri
#ProjectUri = 'https://github.com/gildas/posh-ic'

# License Uri
#LicenseUri = 'https://raw.githubusercontent.com/gildas/posh-ic/master/LICENSE'

# Author of this module
Author = 'Gildas Cherruel'

# Company or vendor of this module
CompanyName = ''

# Copyright statement for this module
Copyright = '(c) 2015 gildas.cherruel. All rights reserved.'

# Description of the functionality provided by this module
Description = 'Powershell module to connect and manage an Interaction Center Server (CIC)'

# Minimum version of the Windows PowerShell engine required by this module
PowerShellVersion = '3.0'

# Name of the Windows PowerShell host required by this module
# PowerShellHostName = ''

# Minimum version of the Windows PowerShell host required by this module
# PowerShellHostVersion = ''

# Minimum version of Microsoft .NET Framework required by this module
# DotNetFrameworkVersion = ''

# Minimum version of the common language runtime (CLR) required by this module
# CLRVersion = ''

# Processor architecture (None, X86, Amd64) required by this module
# ProcessorArchitecture = ''

# Modules that must be imported into the global environment prior to importing this module
# RequiredModules = @()

# Assemblies that must be loaded prior to importing this module
# RequiredAssemblies = @()

# Script files (.ps1) that are run in the caller's environment prior to importing this module.
# ScriptsToProcess = @()

# Type files (.ps1xml) to be loaded when importing this module
# TypesToProcess = @()

# Format files (.ps1xml) to be loaded when importing this module
# FormatsToProcess = @()

# Modules to import as nested modules of the module specified in RootModule/ModuleToProcess
# NestedModules = @()

# Functions to export from this module
FunctionsToExport = @(
    'New-ICSession',
    'Remove-ICSession',
    'Get-ICSessionStatus',
    'Get-ICUserStatus',
    'Get-ICUser',
    'Get-ICUsers',
    'New-ICUser',
    'New-ICUsers',
    'Remove-ICUser',
    'Get-ICWorkgroup',
    'Get-ICWorkgroups',
    'New-ICWorkgroup',
    'New-ICWorkgroups',
    'Remove-ICWorkgroup',
    'New-ICConfigurationId',
    'Get-IPAProcesses',
    'Get-IPAProcess',
    'Start-IPAProcess',
    'Import-IPAProcess',
    'Export-IPAProcess',
    'Get-ICSkills',
    'Get-ICSkill',
    'New-ICSkill',
    'Remove-ICSkill',
    'Get-ICLicenseAllocations',
    'Import-AttProfile',
    'Export-AttProfile',
    'Set-ICLicense'
  )

# Cmdlets to export from this module
CmdletsToExport = ''

# Variables to export from this module
VariablesToExport = ''

# Aliases to export from this module
AliasesToExport = ''

# List of all modules packaged with this module
# ModuleList = @()

# List of all files packaged with this module
# FileList = @()

# Private data to pass to the module specified in RootModule/ModuleToProcess
# PrivateData = ''

# HelpInfo URI of this module
# HelpInfoURI = ''

# Default prefix for commands exported from this module. Override the default prefix using Import-Module -Prefix.
# DefaultCommandPrefix = ''

}
