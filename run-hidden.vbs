' Start WhisperType with no visible console window (runs quietly in the tray).
Set fso = CreateObject("Scripting.FileSystemObject")
appDir = fso.GetParentFolderName(WScript.ScriptFullName)
pyw = appDir & "\.venv\Scripts\pythonw.exe"
script = appDir & "\whispertype.py"
CreateObject("WScript.Shell").Run """" & pyw & """ """ & script & """", 0, False
