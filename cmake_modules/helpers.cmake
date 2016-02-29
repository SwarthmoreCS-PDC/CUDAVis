macro(symlink_data TGT GLOBPAT)
  set(DESTPATH ${CMAKE_CURRENT_BINARY_DIR})
  file(GLOB LNK_FILES
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    ${GLOBPAT})
  if (NOT TARGET ${TGT})
    add_custom_target(${TGT} ALL
      COMMENT "Adding new link target: ${TGT}")
  endif (NOT TARGET ${TGT})

  foreach(FILENAME ${LNK_FILES})
    set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${FILENAME}")
    set(DST "${DESTPATH}/${FILENAME}")
    add_custom_command(
      TARGET ${TGT} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink ${SRC} ${DST}
      )
  endforeach(FILENAME)
endmacro(symlink_data)
