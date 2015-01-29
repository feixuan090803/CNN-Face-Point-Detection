// stdafx.h : include file for standard system include files,
//  or project specific include files that are used frequently, but
//      are changed infrequently
//

#if !defined(AFX_STDAFX_H__2D1DC7F8_E2AD_4912_A01E_8D8F6DC2C988__INCLUDED_)
#define AFX_STDAFX_H__2D1DC7F8_E2AD_4912_A01E_8D8F6DC2C988__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

#define VC_EXTRALEAN		// Exclude rarely-used stuff from Windows headers

#include <afxwin.h>         // MFC core and standard components
#include <afxext.h>         // MFC extensions
#include <afxole.h>         // MFC OLE classes
#include <afxodlgs.h>       // MFC OLE dialog classes
#include <afxdisp.h>        // MFC Automation classes
#include <afxdtctl.h>		// MFC support for Internet Explorer 4 Common Controls
#ifndef _AFX_NO_AFXCMN_SUPPORT
#include <afxcmn.h>			// MFC support for Windows Common Controls
#endif // _AFX_NO_AFXCMN_SUPPORT

#include <afxsock.h>		// MFC socket extensions
#include <iostream>


const UINT g_cImageSize = 39;
const UINT g_cVectorSize = 39;


#define RGB_TO_BGRQUAD(r,g,b) (RGB((b),(g),(r)))

//struct point
//{
//	double x;
//	double y;
//};


//{{AFX_INSERT_LOCATION}}
// Microsoft Visual C++ will insert additional declarations immediately before the previous line.


#endif // !defined(AFX_STDAFX_H__2D1DC7F8_E2AD_4912_A01E_8D8F6DC2C988__INCLUDED_)
