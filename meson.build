project('pattern-recognition-systems', 'cpp', default_options: ['cpp_std=c++17'])

cpp_project_args = ['-std=c++17']
add_project_arguments(cpp_project_args, language: 'cpp')

deps = [
    dependency('opencv4')
]

labs = [
    'Lab1/Lab1.cpp',
    'Lab2/Lab2.cpp',
    'Lab3/Lab3.cpp',
    'Lab4/Lab4.cpp',
    'Lab5/Lab5.cpp',
    'Lab6/Lab6.cpp',
    'Lab7/Lab7.cpp',
    'Lab8/Lab8.cpp',
    'Lab9/Lab9.cpp',
    'LabA/LabA.cpp',
]

foreach lab_src: labs
    lab_name = lab_src.split('/')[0]
    executable(
        lab_name,
        sources: [lab_src],
        include_directories: include_directories(lab_name),
        dependencies: deps,
    )
endforeach
