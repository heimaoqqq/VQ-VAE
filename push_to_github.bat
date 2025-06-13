@echo off
echo 推送代码到GitHub...
git push -u origin master
if %ERRORLEVEL% neq 0 (
  echo 推送失败，请检查网络连接和GitHub认证。
  echo 手动执行: git push -u origin master
) else (
  echo 推送成功！
)
pause 