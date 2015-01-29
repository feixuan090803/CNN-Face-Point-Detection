#include "ReadIniFile.h"
#include "stdafx.h"

extern CPreferences GetPreferences();

CReadIniFile::CReadIniFile(void)
{
}

CReadIniFile theApp;


CReadIniFile::~CReadIniFile(void)
{
}


BOOL CReadIniFile::InitInstance(void)
{
	// initialize path name for current module
	this->m_sModulePath.Empty();
	::GetModuleFileName(NULL,m_sModulePath.GetBuffer(255),255);	   //获取当前进程已加载模块的文件的完整路径
	::PathMakePrettyA(m_sModulePath.GetBuffer(255));			   //转换文件路径名为小写形式
	::PathRemoveFileSpecA(m_sModulePath.GetBuffer(255));		   //得到文件的路径，去掉文件名
	m_sModulePath.ReleaseBuffer();

	free((void*)m_pszProfileName);

	// Next, change the name of the .INI file.
	// The CWinApp destructor will free the memory.
	//得到系统配置文件的路径
	CString tINI = m_sModulePath + _T("\\CNN.ini");
	
	m_pszProfileName=_tcsdup( tINI );

	//////////////////////////////////////////////////////
	//
	// Finally, initialize preferences
	// 读取配置文件，初始化程序配置参数
	//
	//////////////////////////////////////////////////////

	CPreferences::GetPreferences()->ReadParameterFile( this );

	return TRUE;
}
