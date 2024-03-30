# CMakeList.txt : Juerka の CMake プロジェクト。ソースを含めて、次を定義します:
# プロジェクト専用ロジックはこちらです。

find_package(TBB REQUIRED COMPONENTS tbb)

#add_subdirectory(CommonNet)
#add_subdirectory(Main)
#add_subdirectory(Utility)

# ソースをこのプロジェクトの実行可能ファイルに追加します。
#add_executable (Juerka "Juerka.cpp" "Juerka.h")
add_executable (Juerka "CommonNet/Common.h"
	"CommonNet/NetworkGroup.h" "CommonNet/NetworkGroup.cpp"
	"CommonNet/SerialNet.h" "CommonNet/SerialNet.cpp"
	"Main/main.cpp"
	"Utility/Logger.h" "Utility/Logger.cpp"
	"Utility/TimeRecorder.hpp"
 "Utility/WeightLogger.cpp" "Utility/WeightLogger.h")


target_link_libraries(Juerka tbb)

target_include_directories(Juerka PRIVATE "CommonNet")
target_include_directories(Juerka PRIVATE "Main")
target_include_directories(Juerka PRIVATE "Utility")

set_property(TARGET Juerka PROPERTY CXX_STANDARD 20)

#target_link_libraries(Juerka Juerka_CommonNet Juerka_Main Juerka_Utility)

# TODO: テストを追加し、必要な場合は、ターゲットをインストールします。
